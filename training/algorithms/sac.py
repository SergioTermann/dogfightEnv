# -*- coding: utf-8 -*-
"""SAC (Haarnoja et al. 2018) for continuous Box actions: tanh-Gaussian actor,
twin critics with targets, automatic entropy tuning, soft updates."""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from ..buffers import ReplayBuffer
from ..networks import mlp
from .base import BaseAlgorithm

LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0


class SAC(BaseAlgorithm):
    name = "sac"

    DEFAULTS = {
        "hidden_sizes": (256, 256),
        "lr": 3e-4,
        "gamma": 0.99,
        "tau": 0.005,
        "batch_size": 256,
        "buffer_size": 300_000,
        "learning_starts": 5_000,
        "train_freq": 1,
        "target_entropy": "auto",   # "-dim(A)" when auto
        "initial_alpha": 0.2,
        "max_grad_norm": None,
    }

    def __init__(self, obs_dim, action_space, cfg=None, device=None):
        super().__init__(obs_dim, action_space, cfg, device)
        hs = tuple(self.cfg["hidden_sizes"])
        self.act_dim = action_space.shape[0]

        self.actor = mlp([obs_dim, *hs, 2 * self.act_dim])
        self.q1 = mlp([obs_dim + self.act_dim, *hs, 1])
        self.q2 = mlp([obs_dim + self.act_dim, *hs, 1])
        self.q1_target = mlp([obs_dim + self.act_dim, *hs, 1])
        self.q2_target = mlp([obs_dim + self.act_dim, *hs, 1])
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        for p in list(self.q1_target.parameters()) + list(self.q2_target.parameters()):
            p.requires_grad_(False)

        self.log_alpha = torch.tensor(float(np.log(self.cfg["initial_alpha"])),
                                      device=self.device, requires_grad=True)
        if self.cfg["target_entropy"] == "auto":
            self.target_entropy = -float(self.act_dim)
        else:
            self.target_entropy = float(self.cfg["target_entropy"])

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.cfg["lr"])
        self.critic_opt = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=self.cfg["lr"])
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=self.cfg["lr"])
        self.buffer = ReplayBuffer(self.cfg["buffer_size"], obs_dim, self.act_dim)
        self._to(self.device)

    def _to(self, device):
        self.actor.to(device)
        self.q1.to(device); self.q2.to(device)
        self.q1_target.to(device); self.q2_target.to(device)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    # ------------------------------------------------------------------ policy

    def _actor_dist(self, obs_t):
        out = self.actor(obs_t)
        mean, log_std = out.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return Normal(mean, log_std.exp())

    @staticmethod
    def _squash(dist):
        raw = dist.rsample()
        action = torch.tanh(raw)
        # log prob with tanh change-of-variables
        logp = dist.log_prob(raw).sum(-1) - torch.log(
            nn.functional.relu(1 - action.pow(2)) + 1e-6).sum(-1)
        return action, logp

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32)).to(self.device)
        dist = self._actor_dist(obs_t)
        if deterministic:
            action = torch.tanh(dist.mean)
        else:
            action, _ = self._squash(dist)
        # unclipped: the adapter clips to the env bounds on env.step
        return action.cpu().numpy()

    def observe(self, obs, action, reward, next_obs, done):
        self.buffer.add(obs, action, reward, next_obs, done)

    # ------------------------------------------------------------------ update

    def update(self):
        if len(self.buffer) < self.cfg["learning_starts"]:
            return {}
        obs, act, rew, next_obs, done = self.buffer.sample(self.cfg["batch_size"], self.device)

        with torch.no_grad():
            next_a, next_logp = self._squash(self._actor_dist(next_obs))
            q_next = torch.min(self.q1_target(torch.cat([next_obs, next_a], -1)),
                               self.q2_target(torch.cat([next_obs, next_a], -1))).flatten()
            target_q = rew + self.cfg["gamma"] * (1 - done) * (q_next - self.alpha * next_logp)

        q1_val = self.q1(torch.cat([obs, act], -1)).flatten()
        q2_val = self.q2(torch.cat([obs, act], -1)).flatten()
        q_loss = nn.functional.mse_loss(q1_val, target_q) + nn.functional.mse_loss(q2_val, target_q)
        self.critic_opt.zero_grad()
        q_loss.backward()
        if self.cfg["max_grad_norm"]:
            nn.utils.clip_grad_norm_(list(self.q1.parameters()) + list(self.q2.parameters()),
                                     self.cfg["max_grad_norm"])
        self.critic_opt.step()

        a_new, logp = self._squash(self._actor_dist(obs))
        q_new = torch.min(self.q1(torch.cat([obs, a_new], -1)),
                          self.q2(torch.cat([obs, a_new], -1))).flatten()
        pi_loss = (self.alpha.detach() * logp - q_new).mean()
        self.actor_opt.zero_grad()
        pi_loss.backward()
        if self.cfg["max_grad_norm"]:
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg["max_grad_norm"])
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        with torch.no_grad():
            tau = self.cfg["tau"]
            for p, pt in zip(self.q1.parameters(), self.q1_target.parameters()):
                pt.mul_(1 - tau).add_(p, alpha=tau)
            for p, pt in zip(self.q2.parameters(), self.q2_target.parameters()):
                pt.mul_(1 - tau).add_(p, alpha=tau)

        return {"q_loss": q_loss.item(), "pi_loss": pi_loss.item(),
                "alpha": self.alpha.item(), "alpha_loss": alpha_loss.item()}

    # ------------------------------------------------------------------ io

    def save(self, path):
        self._ensure_dir(path)
        payload = super().save(path)
        payload["act_dim"] = self.act_dim
        payload["act_low"] = np.asarray(self.action_space.low, dtype=np.float32).tolist()
        payload["act_high"] = np.asarray(self.action_space.high, dtype=np.float32).tolist()
        payload["state"] = {
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "q1_target": self.q1_target.state_dict(),
            "q2_target": self.q2_target.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "alpha_opt": self.alpha_opt.state_dict(),
        }
        torch.save(payload, path)

    @staticmethod
    def load(path, device=None):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        from .ppo import _space_from
        algo = SAC(payload["obs_dim"], _space_from(payload), payload["cfg"], device)
        st = payload["state"]
        algo.actor.load_state_dict(st["actor"])
        algo.q1.load_state_dict(st["q1"])
        algo.q2.load_state_dict(st["q2"])
        algo.q1_target.load_state_dict(st["q1_target"])
        algo.q2_target.load_state_dict(st["q2_target"])
        algo.log_alpha.data = st["log_alpha"].to(algo.device)
        algo.actor_opt.load_state_dict(st["actor_opt"])
        algo.critic_opt.load_state_dict(st["critic_opt"])
        algo.alpha_opt.load_state_dict(st["alpha_opt"])
        return algo
