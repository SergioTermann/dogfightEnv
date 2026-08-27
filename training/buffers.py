# -*- coding: utf-8 -*-
"""Experience buffers: PPO rollout storage, uniform replay (SAC) and
prioritized replay with n-step reconstruction (Rainbow)."""

import numpy as np
import torch


class RolloutBuffer:
    """Fixed-size on-policy storage for one PPO iteration."""

    def __init__(self, size, obs_dim, act_dim, gamma, gae_lambda):
        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((size, act_dim), dtype=np.float32)
        self.logp = np.zeros(size, dtype=np.float32)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.dones = np.zeros(size, dtype=np.float32)
        self.values = np.zeros(size, dtype=np.float32)
        self.advantages = np.zeros(size, dtype=np.float32)
        self.returns = np.zeros(size, dtype=np.float32)
        self.gamma, self.gae_lambda = gamma, gae_lambda
        self.ptr = 0
        self.size = size

    def add(self, obs, action, logp, value, reward, done):
        i = self.ptr
        self.obs[i] = obs
        self.actions[i] = action
        self.logp[i] = logp
        self.values[i] = value
        self.rewards[i] = reward
        self.dones[i] = float(done)
        self.ptr += 1

    def finish(self, last_value):
        """GAE advantages and returns; call once the buffer is full."""
        gae = 0.0
        for i in reversed(range(self.ptr)):
            next_value = last_value if i == self.ptr - 1 else self.values[i + 1]
            non_terminal = 1.0 - self.dones[i]
            delta = self.rewards[i] + self.gamma * next_value * non_terminal - self.values[i]
            gae = delta + self.gamma * self.gae_lambda * non_terminal * gae
            self.advantages[i] = gae
        self.returns[: self.ptr] = self.advantages[: self.ptr] + self.values[: self.ptr]

    def get(self, device, batch_size, shuffle=True):
        """Yield shuffled minibatches as tensors."""
        n = self.ptr
        indices = np.arange(n)
        if shuffle:
            np.random.shuffle(indices)
        for start in range(0, n, batch_size):
            idx = indices[start : start + batch_size]
            yield (
                torch.as_tensor(self.obs[idx]).to(device),
                torch.as_tensor(self.actions[idx]).to(device),
                torch.as_tensor(self.logp[idx]).to(device),
                torch.as_tensor(self.advantages[idx]).to(device),
                torch.as_tensor(self.returns[idx]).to(device),
            )


class ReplayBuffer:
    """Uniform replay buffer with numpy storage (SAC)."""

    def __init__(self, capacity, obs_dim, act_dim):
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.capacity = capacity
        self.ptr = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.ptr

    def add(self, obs, action, reward, next_obs, done):
        i = self.ptr
        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_obs[i] = next_obs
        self.dones[i] = float(done)
        self.ptr = (i + 1) % self.capacity
        if self.ptr == 0:
            self.full = True

    def sample(self, batch_size, device):
        idx = np.random.randint(0, len(self), size=batch_size)
        return (
            torch.as_tensor(self.obs[idx]).to(device),
            torch.as_tensor(self.actions[idx]).to(device),
            torch.as_tensor(self.rewards[idx]).to(device),
            torch.as_tensor(self.next_obs[idx]).to(device),
            torch.as_tensor(self.dones[idx]).to(device),
        )


class PrioritizedReplayBuffer(ReplayBuffer):
    """Proportional prioritized replay (PER, Schaul et al. 2016) with n-step
    return reconstruction at sample time (Rainbow).

    The n-step window [i, i+n-1] is only valid when none of the transitions
    i..i+n-2 ends an episode and the ring pointer never overwrote a slot the
    window reaches into; invalid indices are excluded from sampling.
    """

    def __init__(self, capacity, obs_dim, n_action_dims=1,
                 alpha=0.6, beta_start=0.4, beta_frames=1_000_000):
        super().__init__(capacity, obs_dim, n_action_dims)
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.alpha = alpha
        self.beta = beta_start
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self._max_priority = 1.0
        self._frame = 0

    def add(self, obs, action, reward, next_obs, done):
        super().add(obs, action, reward, next_obs, done)
        self.priorities[(self.ptr - 1) % self.capacity] = self._max_priority

    def _valid_indices(self, n_step):
        n = len(self)
        idx = np.arange(n)
        ok = np.ones(n, dtype=bool)
        newest = (self.ptr - 1) % self.capacity
        for k in range(n_step - 1):
            j = (idx + k) % self.capacity
            ok &= self.dones[j] == 0.0        # no termination inside the window
            ok &= j != newest                 # time-successor must already exist
        if not ok.any():
            return np.arange(n)
        return np.where(ok)[0]

    def sample(self, batch_size, device, n_step=1, gamma=0.99):
        self._frame += 1
        frac = min(1.0, self._frame / max(1, self.beta_frames))
        self.beta = self.beta_start + (1.0 - self.beta_start) * frac

        valid = self._valid_indices(n_step)
        probs = self.priorities[valid] ** self.alpha
        total = probs.sum()
        probs = probs / total if total > 0 else np.full(len(valid), 1.0 / len(valid), dtype=np.float32)

        if len(valid) >= batch_size:
            chosen = np.random.choice(valid, size=batch_size, p=probs, replace=False)
        else:
            chosen = valid[np.random.randint(0, len(valid), size=batch_size)]
        p_sample = probs[np.searchsorted(valid, chosen)]
        weights = (len(valid) * p_sample) ** (-self.beta)
        weights = (weights / weights.max()).astype(np.float32)

        obs = self.obs[chosen]
        next_obs = self.next_obs[chosen]
        actions = self.actions[chosen]
        rewards = self.rewards[chosen].copy()
        dones = self.dones[chosen].copy()
        if n_step > 1:
            gamma_pow = gamma
            for k in range(1, n_step):
                nk = (chosen + k) % self.capacity
                rewards += gamma_pow * self.rewards[nk]
                dones = np.maximum(dones, self.dones[nk])
                gamma_pow *= gamma
            # bootstrap from the state at the END of the n-step window; when an
            # intermediate episode ended, valid_indices guaranteed dones=1 and
            # the extra rewards all belong to a new episode with done=1 ->
            # bootstrap zeroed, which is still a usable (slightly mixed) sample
            next_obs = self.next_obs[(chosen + n_step - 1) % self.capacity]
        return (
            torch.as_tensor(obs).to(device),
            torch.as_tensor(actions).to(device),
            torch.as_tensor(rewards).to(device),
            torch.as_tensor(next_obs).to(device),
            torch.as_tensor(dones).to(device),
            torch.as_tensor(weights).to(device),
            torch.as_tensor(chosen).to(device),
        )

    def update_priorities(self, indices, priorities):
        priorities = np.asarray(priorities, dtype=np.float32)
        self.priorities[indices] = priorities
        self._max_priority = max(self._max_priority, float(priorities.max()))
