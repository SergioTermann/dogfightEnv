# -*- coding: utf-8 -*-
"""PPO (Schulman et al. 2017) for continuous Box actions: Gaussian policy with
state-independent log-std, GAE advantages, clipped surrogate objective."""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from ..buffers import RolloutBuffer
from ..networks import mlp
from .base import BaseAlgorithm


class PPO(BaseAlgorithm):
    name = "ppo"

    DEFAULTS = {
        "hidden_sizes": (128, 128),
        "lr": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "rollout_size": 2048,
        "batch_size": 256,
        "epochs": 10,
        "entropy_coef": 0.003,
        "value_coef": 0.5,
        "max_grad_norm": 0.5,
        "log_std_init": 0.0,
    }

    def __init__(self, obs_dim, action_space, cfg=None, device=None):
        super().__init__(obs_dim, action_space, cfg, device)
        hs = tuple(self.cfg["hidden_sizes"])
        self.act_dim = action_space.shape[0]
        self.policy_net = mlp([obs_dim, *hs, self.act_dim], out_activation=nn.Tanh)
        self.value_net = mlp([obs_dim, *hs, 1])
        self.log_std = nn.Parameter(torch.full((self.act_dim,), float(self.cfg["log_std_init"])))
        self.optimizer = torch.optim.Adam(
            list(self.policy_net.parameters()) + list(self.value_net.parameters())
            + [self.log_std], lr=self.cfg["lr"])
        self.buffer = RolloutBuffer(self.cfg["rollout_size"], obs_dim, self.act_dim,
                                    self.cfg["gamma"], self.cfg["gae_lambda"])
        self.to(self.device)

    def to(self, device):
        self.policy_net.to(device)
        self.value_net.to(device)
        self.log_std.data = self.log_std.data.to(device)

    def _dist(self, obs_t):
        mean = self.policy_net(obs_t)
        return Normal(mean, self.log_std.exp().expand_as(mean))

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32)).to(self.device)
        dist = self._dist(obs_t)
        if deterministic:
            action = dist.mean
        else:
            action = dist.sample()
        logp = dist.log_prob(action).sum(-1)
        value = self.value_net(obs_t).flatten()
        # the RAW sample goes into the rollout buffer so logp matches the
        # stored action; the adapter clips to the env bounds on env.step
        return (action.cpu().numpy(),
                float(logp.item()),
                float(value.item()))

    def observe(self, obs, action, logp, value, reward, done):
        self.buffer.add(obs, action, logp, value, reward, done)

    @torch.no_grad()
    def value_of(self, obs):
        """V(s) for the bootstrap of a truncated rollout."""
        obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32)).to(self.device)
        return float(self.value_net(obs_t).flatten().item())

    def update(self, last_value):
        """Consume the full rollout; returns loss stats."""
        buf = self.buffer
        if buf.ptr == 0:
            return {}
        buf.finish(last_value)
        stats = {"pi_loss": 0.0, "v_loss": 0.0, "entropy": 0.0, "kl": 0.0}
        n_batches = 0
        for _ in range(self.cfg["epochs"]):
            for obs_t, act_t, old_logp, adv, ret in buf.get(self.device, self.cfg["batch_size"]):
                dist = self._dist(obs_t)
                logp = dist.log_prob(act_t).sum(-1)
                entropy = dist.entropy().sum(-1).mean()
                ratio = (logp - old_logp).exp()

                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - self.cfg["clip_range"], 1 + self.cfg["clip_range"]) * adv
                pi_loss = -torch.min(surr1, surr2).mean()

                value = self.value_net(obs_t).flatten()
                v_loss = nn.functional.mse_loss(value, ret)

                loss = pi_loss + self.cfg["value_coef"] * v_loss - self.cfg["entropy_coef"] * entropy
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.policy_net.parameters()) + list(self.value_net.parameters()) + [self.log_std],
                    self.cfg["max_grad_norm"])
                self.optimizer.step()

                with torch.no_grad():
                    kl = (old_logp - logp).mean()
                stats["pi_loss"] += pi_loss.item()
                stats["v_loss"] += v_loss.item()
                stats["entropy"] += entropy.item()
                stats["kl"] += kl.item()
                n_batches += 1
        for k in stats:
            stats[k] /= max(1, n_batches)
        buf.ptr = 0
        return stats

    # ------------------------------------------------------------------ io

    def save(self, path):
        self._ensure_dir(path)
        payload = super().save(path)
        payload["act_dim"] = self.act_dim
        payload["act_low"] = np.asarray(self.action_space.low, dtype=np.float32).tolist()
        payload["act_high"] = np.asarray(self.action_space.high, dtype=np.float32).tolist()
        payload["state"] = {
            "policy": self.policy_net.state_dict(),
            "value": self.value_net.state_dict(),
            "log_std": self.log_std.detach().cpu(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(payload, path)

    @staticmethod
    def load(path, device=None):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        algo = PPO(payload["obs_dim"], _space_from(payload), payload["cfg"], device)
        st = payload["state"]
        algo.policy_net.load_state_dict(st["policy"])
        algo.value_net.load_state_dict(st["value"])
        algo.log_std.data = st["log_std"].to(algo.device)
        algo.optimizer.load_state_dict(st["optimizer"])
        algo.to(algo.device)
        return algo


def _space_from(payload):
    """Reconstruct a minimal Box-like from stored bounds."""
    import numpy as _np

    class _Box:
        def __init__(self, n, low, high):
            self.shape = (n,)
            self.low = _np.asarray(low, dtype=_np.float32)
            self.high = _np.asarray(high, dtype=_np.float32)

    return _Box(payload["act_dim"], payload.get("act_low", [-1] * payload["act_dim"]),
                payload.get("act_high", [1] * payload["act_dim"]))
