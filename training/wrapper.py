# -*- coding: utf-8 -*-
"""Environment adapters for training.

The repo's envs use the OLD gym API (4-tuple step, reset() -> obs list, no
seed/info). EnvAdapter normalizes that plus:

- float32 observations (they return plain python lists)
- optional running mean/std normalization of observations
- a stale-reset fix: oneVSoneEnv.reset() builds the observation BEFORE
  teleporting the planes, so the first obs describes the previous episode.
  We step once with a neutral action and use that observation instead.

DiscreteActionEnv maps an integer action (for Rainbow) onto a configurable
grid of continuous action vectors (default: roll x pitch x thrust x fire).
"""

import numpy as np


class RunningNorm:
    """Online observation normalizer (mean/var, Welford-style)."""

    def __init__(self, dim, clip=10.0):
        self.mean = np.zeros(dim, dtype=np.float32)
        self.var = np.ones(dim, dtype=np.float32)
        self.count = 1e-4
        self.clip = clip

    def update(self, obs):
        self.count += 1
        delta = obs - self.mean
        self.mean += delta / self.count
        self.var += delta * (obs - self.mean) - self.var / self.count

    def __call__(self, obs):
        return np.clip((obs - self.mean) / np.sqrt(self.var + 1e-8), -self.clip, self.clip).astype(np.float32)

    def state_dict(self):
        return {"mean": self.mean.tolist(), "var": self.var.tolist(), "count": float(self.count)}

    def load_state_dict(self, st):
        self.mean = np.asarray(st["mean"], dtype=np.float32)
        self.var = np.asarray(st["var"], dtype=np.float32)
        self.count = float(st["count"])


class EnvAdapter:
    """Old-gym env -> (obs float32 ndarray, reward, done, info) interface."""

    def __init__(self, env, normalize=True, fix_stale_reset=True, update_norm=True):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.obs_dim = int(np.prod(env.observation_space.shape))
        self.normalize = normalize
        if normalize:
            self.norm = RunningNorm(self.obs_dim)
        else:
            self.norm = None
        self._fix_stale_reset = fix_stale_reset
        self.update_norm = update_norm
        self._neutral = self._neutral_action()

    def _neutral_action(self):
        low = np.asarray(self.action_space.low, dtype=np.float32)
        high = np.asarray(self.action_space.high, dtype=np.float32)
        mid = (low + high) / 2.0
        # fire channel: keep at "no fire" (low), thrust in the middle
        if hasattr(self.action_space, "low") and mid.shape[-1] in (5, 7):
            mid[-1] = low[-1]
        return mid

    def _proc(self, obs, update=True):
        obs = np.asarray(obs, dtype=np.float32).flatten()
        if self.normalize and self.norm is not None:
            if update and self.update_norm:
                self.norm.update(obs)
            return self.norm(obs)
        return obs

    def reset(self):
        obs = self.env.reset()
        if self._fix_stale_reset:
            # one neutral step -> observation that matches the fresh spawn
            obs, _, done, _ = self.env.step(self._neutral)
            if done:
                obs = self.env.reset()
        return self._proc(obs)

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32),
                         self.action_space.low, self.action_space.high)
        obs, reward, done, info = self.env.step(action)
        return self._proc(obs), float(reward), bool(done), (info or {})

    def close(self):
        close = getattr(self.env, "close", None)
        if close:
            close()

    # passthrough for env-specific attributes (e.g. Tacview log paths)
    def __getattr__(self, name):
        return getattr(self.env, name)


class DiscreteActionEnv:
    """Wraps a continuous env so it looks discrete for value-based algos.

    action index = ((i_roll * n_pitch + i_pitch) * n_thrust + i_thrust) * n_fire + i_fire
    """

    DEFAULT_GRID = {
        "roll": [-1.0, 0.0, 1.0],
        "pitch": [-1.0, 0.0, 1.0],
        "yaw": [0.0],
        "thrust": [0.4, 0.7, 1.0],
        "fire": [0.0, 1.0],
    }

    def __init__(self, adapter, grid=None):
        self.adapter = adapter
        cont = adapter.action_space
        self.act_dim = int(np.prod(cont.shape))
        grid = grid or self.DEFAULT_GRID
        if self.act_dim == 5:
            axes = [grid["roll"], grid["pitch"], grid["yaw"], grid["thrust"], grid["fire"]]
        elif self.act_dim == 7:  # IA_enemy_env declares 7 dims
            axes = [grid["roll"], grid["pitch"], grid["yaw"], [0.0], [0.0],
                    grid["thrust"], grid["fire"]]
        else:
            axes = [[0.0]] * self.act_dim
            for i in range(min(3, self.act_dim)):
                axes[i] = grid["roll"]
        self.actions = np.array([self._combine(a) for a in self._product(*axes)], dtype=np.float32)
        self.n = len(self.actions)

        class _Disc:
            def __init__(self, n):
                self.n = n

        self.action_space = _Disc(self.n)

    @staticmethod
    def _product(*axes):
        out = [[]]
        for axis in axes:
            out = [row + [v] for row in out for v in axis]
        return out

    def _combine(self, combo):
        return combo

    def reset(self):
        return self.adapter.reset()

    def step(self, index):
        return self.adapter.step(self.actions[int(index)])

    @property
    def norm(self):
        return self.adapter.norm

    @property
    def obs_dim(self):
        return self.adapter.obs_dim
