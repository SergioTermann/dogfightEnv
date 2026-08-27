# -*- coding: utf-8 -*-
"""Algorithm unit tests on a toy task that mimics the dogfight env shape
(26-dim obs, 5-dim continuous action, old 4-tuple gym API) - no sandbox needed.

Run:  python -m training.tests.test_algos_toy
"""

import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for p in (ROOT,):
    if p not in sys.path:
        sys.path.insert(0, p)

from training.algorithms import create_algo, load_algo  # noqa: E402
from training.wrapper import DiscreteActionEnv, EnvAdapter  # noqa: E402


class ToyBox:
    def __init__(self, low, high):
        self.low = np.asarray(low, dtype=np.float32)
        self.high = np.asarray(high, dtype=np.float32)
        self.shape = self.low.shape


class ToyEnv:
    """1-D 'aim' task dressed up like oneVSoneEnv: steer action[1] (pitch) to
    track a drifting target value hidden in obs[0]. Reward = -|error| (+bonus
    under threshold). 26-dim obs padded with noise, episodes 50 steps."""

    def __init__(self):
        self.action_space = ToyBox([-1, -1, -1, 0, 0], [1, 1, 1, 1, 1])
        self.observation_space = ToyBox([-1] * 26, [1] * 26)
        self.t = 0
        self.angle = 0.0
        self.target = 0.0

    def _obs(self):
        base = np.zeros(26, dtype=np.float32)
        base[0] = self.target
        base[1] = self.angle
        # padding dims stay constant: the test targets the algorithms, not
        # noise robustness (the real env's 26 dims all carry signal)
        return list(base)

    def reset(self):
        self.t = 0
        self.angle = np.random.uniform(-1, 1)
        self.target = np.random.uniform(-1, 1)
        return self._obs()

    def step(self, action):
        self.t += 1
        self.angle = np.clip(self.angle + 0.25 * float(np.clip(action[1], -1, 1)), -1, 1)
        err = abs(self.angle - self.target)
        reward = -err + (1.0 if err < 0.15 else 0.0)
        done = self.t >= 50
        return self._obs(), reward, done, {}


def run_episodes(algo, adapter, n=5, deterministic=False):
    rets = []
    for _ in range(n):
        obs = adapter.reset()
        done, total = False, 0.0
        while not done:
            out = algo.act(obs, deterministic=deterministic)
            action = out[0] if algo.name == "ppo" else out
            obs, r, done, _ = adapter.step(action)
            total += r
        rets.append(total)
    return float(np.mean(rets))


def train_ppo(steps=30000):
    env = ToyEnv()
    adapter = EnvAdapter(env)
    algo = create_algo("ppo", adapter.obs_dim, adapter.action_space,
                       {"rollout_size": 512, "batch_size": 128, "epochs": 6,
                        "hidden_sizes": (64, 64), "lr": 1e-3, "entropy_coef": 0.001})
    baseline = run_episodes(algo, adapter, n=3)
    obs = adapter.reset()
    done = False
    for t in range(steps):
        action, logp, value = algo.act(obs)
        next_obs, r, done, _ = adapter.step(action)
        algo.observe(obs, action, logp, value, r, done)
        if algo.buffer.ptr >= algo.buffer.size:
            algo.update(algo.value_of(next_obs))
        obs = adapter.reset() if done else next_obs
    final = run_episodes(algo, adapter, n=3)
    return baseline, final, algo, adapter


def train_sac(steps=15000):
    env = ToyEnv()
    adapter = EnvAdapter(env)
    algo = create_algo("sac", adapter.obs_dim, adapter.action_space,
                       {"hidden_sizes": (64, 64), "learning_starts": 1000,
                        "batch_size": 256, "lr": 3e-4, "target_entropy": -3.0})
    baseline = run_episodes(algo, adapter, n=3)
    obs = adapter.reset()
    done = False
    for t in range(steps):
        action = algo.act(obs)
        next_obs, r, done, _ = adapter.step(action)
        algo.observe(obs, action, r, next_obs, done)
        algo.update()
        obs = adapter.reset() if done else next_obs
    final = run_episodes(algo, adapter, n=3)
    return baseline, final, algo, adapter


def train_rainbow(steps=12000):
    env = ToyEnv()
    adapter = DiscreteActionEnv(EnvAdapter(env),
                                {"roll": [0.0], "pitch": [-1.0, 0.0, 1.0], "yaw": [0.0],
                                 "thrust": [0.7], "fire": [0.0]})
    algo = create_algo("rainbow", adapter.obs_dim, adapter.action_space,
                       {"hidden_sizes": (64, 64), "learning_starts": 300,
                        "batch_size": 32, "lr": 1e-3, "n_step": 3,
                        "v_min": -10.0, "v_max": 10.0,
                        "target_update_interval": 300})
    baseline = run_episodes(algo, adapter, n=3)
    obs = adapter.reset()
    done = False
    for t in range(steps):
        action = algo.act(obs)
        next_obs, r, done, _ = adapter.step(action)
        algo.observe(obs, action, r, next_obs, done)
        algo.update()
        obs = adapter.reset() if done else next_obs
    final = run_episodes(algo, adapter, n=3)
    return baseline, final, algo, adapter


def main():
    results = []

    def check(name, ok, detail=""):
        results.append((name, ok))
        print("%-40s %s  %s" % (name, "PASS" if ok else "FAIL", detail))

    for trainer, budget in ((train_ppo, 30000), (train_sac, 15000), (train_rainbow, 12000)):
        algo_name = {train_ppo: "ppo", train_sac: "sac", train_rainbow: "rainbow"}[trainer]
        base, final, algo, adapter = trainer(budget)
        check("%s: improves on toy task" % algo_name, final > base + 5.0,
              "baseline %.1f -> final %.1f" % (base, final))

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "%s.pt" % algo_name)
            algo.save(path)
            loaded = load_algo(path)
            def _action(a):
                return np.asarray(a[0] if algo.name == "ppo" else a, dtype=np.float32)
            fixed_obs = adapter.reset()   # SAME observation for both networks
            a1 = _action(algo.act(fixed_obs, deterministic=True))
            a2 = _action(loaded.act(fixed_obs, deterministic=True))
            same = np.allclose(np.asarray(a1, dtype=np.float32),
                               np.asarray(a2, dtype=np.float32), atol=1e-4)
            check("%s: save/load roundtrip" % algo_name, same,
                  "%r vs %r" % (a1, a2))

    failed = [n for n, ok in results if not ok]
    print("\n%d/%d checks passed" % (len(results) - len(failed), len(results)))
    if failed:
        print("FAILED:", failed)
        sys.exit(1)


if __name__ == "__main__":
    main()
