# -*- coding: utf-8 -*-
"""Rainbow DQN (Hessel et al. 2018) for DISCRETE actions.

All six Rainbow components:
  1. n-step returns            (reconstructed at sample time in the PER buffer)
  2. double Q-learning         (online net picks the target action)
  3. dueling networks          (V / A streams)
  4. noisy nets                (exploration, no epsilon-greedy)
  5. prioritized replay        (proportional PER)
  6. distributional RL (C51)   (51-atom categorical value distribution)
"""

import numpy as np
import torch
import torch.nn as nn

from ..buffers import PrioritizedReplayBuffer
from ..networks import NoisyLinear, mlp
from .base import BaseAlgorithm


class RainbowQNetwork(nn.Module):
    """Distributional dueling noisy Q-network: outputs (B, n_actions, atoms) logits."""

    def __init__(self, obs_dim, n_actions, hidden_sizes=(128, 128), atoms=51,
                 v_min=-120.0, v_max=120.0, sigma_init=0.5):
        super().__init__()
        self.atoms = atoms
        self.register_buffer("z_support", torch.linspace(v_min, v_max, atoms))
        hs = list(hidden_sizes)
        self.feature = mlp([obs_dim, *hs])
        self.value_stream = nn.Sequential(
            NoisyLinear(hs[-1], hs[-1] // 2, sigma_init),
            nn.ReLU(),
            NoisyLinear(hs[-1] // 2, atoms, sigma_init),
        )
        self.advantage_stream = nn.Sequential(
            NoisyLinear(hs[-1], hs[-1] // 2, sigma_init),
            nn.ReLU(),
            NoisyLinear(hs[-1] // 2, n_actions * atoms, sigma_init),
        )

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()

    def forward(self, obs):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        feat = self.feature(obs)
        value = self.value_stream(feat).view(feat.size(0), 1, self.atoms)
        adv = self.advantage_stream(feat).view(feat.size(0), -1, self.atoms)
        logits = value + adv - adv.mean(dim=1, keepdim=True)
        return logits  # (B, n_actions, atoms) — unnormalized log-probabilities

    @torch.no_grad()
    def q_values(self, obs):
        """Expected Q per action for action selection (mean over the distribution)."""
        return torch.softmax(self(obs), dim=-1).mul(self.z_support).sum(-1)


class Rainbow(BaseAlgorithm):
    name = "rainbow"

    DEFAULTS = {
        "hidden_sizes": (128, 128),
        "lr": 6.25e-4,
        "gamma": 0.99,
        "n_step": 3,
        "atoms": 51,
        "v_min": -120.0,          # reward range is roughly [-100, +100] in the dogfight envs
        "v_max": 120.0,
        "batch_size": 64,
        "buffer_size": 300_000,
        "learning_starts": 2_000,
        "train_freq": 1,
        "target_update_interval": 2_000,   # hard copy (standard Rainbow)
        "per_alpha": 0.6,
        "per_beta_start": 0.4,
        "per_beta_frames": 1_000_000,
        "noisy_sigma": 0.5,
        "max_grad_norm": 10.0,
    }

    def __init__(self, obs_dim, action_space, cfg=None, device=None):
        super().__init__(obs_dim, action_space, cfg, device)
        self.n_actions = int(action_space.n)
        atoms = self.cfg["atoms"]
        self.atoms = atoms
        self.v_min, self.v_max = self.cfg["v_min"], self.cfg["v_max"]
        self.z_support = torch.linspace(self.v_min, self.v_max, atoms, device=self.device)

        mk = lambda: RainbowQNetwork(obs_dim, self.n_actions,
                                     tuple(self.cfg["hidden_sizes"]), atoms,
                                     self.cfg["v_min"], self.cfg["v_max"], self.cfg["noisy_sigma"])
        self.net = mk()
        self.target_net = mk()
        self.target_net.load_state_dict(self.net.state_dict())
        for p in self.target_net.parameters():
            p.requires_grad_(False)
        self.net.to(self.device)
        self.target_net.to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.cfg["lr"])
        self.buffer = PrioritizedReplayBuffer(
            self.cfg["buffer_size"], obs_dim, n_action_dims=1,
            alpha=self.cfg["per_alpha"], beta_start=self.cfg["per_beta_start"],
            beta_frames=self.cfg["per_beta_frames"])
        self._updates = 0

    # ------------------------------------------------------------------ policy

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        """NoisyNet exploration: greedy w.r.t. the noisy network's expectations.
        `deterministic=True` disables noise entirely (pure greedy, for eval)."""
        if deterministic:
            self.net.eval()
        obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32)).to(self.device)
        q = self.net.q_values(obs_t)
        action = int(torch.argmax(q).item())
        if deterministic:
            self.net.train()
        return action

    def observe(self, obs, action, reward, next_obs, done):
        self.buffer.add(obs, np.asarray([action], dtype=np.float32), reward, next_obs, done)

    # ------------------------------------------------------------------ update

    def _project(self, tz, probs):
        """Distribute target atom positions `tz` (B, atoms) weighted by `probs`
        (B, atoms) onto the fixed support -> categorical target distribution."""
        batch, atoms = tz.shape
        delta = (self.v_max - self.v_min) / (atoms - 1)
        tz = tz.clamp(self.v_min, self.v_max)
        b = (tz - self.v_min) / delta                       # fractional bucket index
        lo, hi = b.floor().long(), b.ceil().long()
        hi = torch.where(lo == hi, hi + 1, hi)              # tz==v_max -> last bucket
        m = torch.zeros_like(tz)
        m.scatter_add_(1, lo, probs * (hi - b))
        m.scatter_add_(1, hi.clamp(max=atoms - 1), probs * (b - lo) * (hi.clamp(max=atoms - 1) == hi))
        return m

    def update(self):
        if len(self.buffer) < self.cfg["learning_starts"]:
            return {}
        self.net.reset_noise()
        self.target_net.reset_noise()

        batch = self.buffer.sample(self.cfg["batch_size"], self.device,
                                    n_step=self.cfg["n_step"], gamma=self.cfg["gamma"])
        obs, act, rew, next_obs, done, weights, indices = batch
        act = act.long().flatten()

        with torch.no_grad():
            # double-Q: ONLINE net chooses the action, TARGET net evaluates it
            next_q = self.net.q_values(next_obs)
            best = next_q.argmax(dim=1)
            target_logits = self.target_net(next_obs)
            target_dist = torch.softmax(target_logits, dim=-1)
            target_dist = target_dist[torch.arange(len(best), device=self.device), best]  # (B, atoms)
            gamma_n = self.cfg["gamma"] ** self.cfg["n_step"]
            # Tz = r + gamma^n * (1-done) * z, each support atom's probability
            # split onto neighbouring buckets by the projection
            tz = rew.unsqueeze(-1) + gamma_n * (1 - done).unsqueeze(-1) * self.z_support.unsqueeze(0)
            m = self._project(tz, target_dist)

        logits = self.net(obs)                                   # (B, A, atoms)
        logq = torch.log_softmax(logits, dim=-1)
        logq_a = logq[torch.arange(len(act), device=self.device), act]  # (B, atoms)
        losses = -(m * logq_a).sum(-1)                           # cross-entropy per sample
        loss = (weights * losses).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg["max_grad_norm"])
        self.optimizer.step()

        self.buffer.update_priorities(indices.cpu().numpy(),
                                      (losses.detach().abs() + 1e-5).cpu().numpy())

        self._updates += 1
        if self._updates % self.cfg["target_update_interval"] == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        return {"loss": loss.item(), "max_q": float(self.net.q_values(obs).max().item())}

    # ------------------------------------------------------------------ io

    def save(self, path):
        self._ensure_dir(path)
        payload = super().save(path)
        payload["n_actions"] = self.n_actions
        payload["state"] = {
            "net": self.net.state_dict(),
            "target": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(payload, path)

    @staticmethod
    def load(path, device=None):
        payload = torch.load(path, map_location="cpu", weights_only=False)

        class _Disc:
            def __init__(self, n):
                self.n = n

        algo = Rainbow(payload["obs_dim"], _Disc(payload["n_actions"]), payload["cfg"], device)
        st = payload["state"]
        algo.net.load_state_dict(st["net"])
        algo.target_net.load_state_dict(st["target"])
        algo.optimizer.load_state_dict(st["optimizer"])
        return algo
