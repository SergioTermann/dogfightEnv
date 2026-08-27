# -*- coding: utf-8 -*-
"""Load a trained checkpoint and watch it fly (rendered 3D view).

Usage:
  python -m training.enjoy --model checkpoints/ppo_oneVSone/model_best.pt --episodes 3
"""

import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from training.algorithms import load_algo  # noqa: E402
from training.train import ENVS, register_envs  # noqa: E402
from training.wrapper import DiscreteActionEnv, EnvAdapter, RunningNorm  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--env", default=None, help="defaults to the env recorded in the checkpoint")
    ap.add_argument("--host", default="192.168.1.103")
    ap.add_argument("--port", default="50888")
    ap.add_argument("--episodes", type=int, default=3)
    args = ap.parse_args()

    algo = load_algo(args.model, device="cpu")
    run_dir = os.path.dirname(args.model)
    tag = os.path.splitext(os.path.basename(args.model))[0].split("_", 1)[1]
    extra_path = os.path.join(run_dir, "extra_%s.pt" % tag)
    extra = torch.load(extra_path, map_location="cpu", weights_only=False) if os.path.exists(extra_path) else {}
    env_name = args.env or extra.get("env", "oneVSone")

    register_envs()
    env = ENVS[env_name](host=args.host, port=args.port, rendering=True)
    adapter = EnvAdapter(env, normalize=False)  # norm stats restored below
    if algo.name == "rainbow":
        adapter = DiscreteActionEnv(adapter)
    if extra.get("norm"):
        adapter.norm = RunningNorm(adapter.obs_dim)
        adapter.norm.load_state_dict(extra["norm"])
        adapter.normalize = True

    print("watching %s (%s) for %d episodes - close the sandbox window to stop"
          % (args.model, algo.name, args.episodes))
    for ep in range(args.episodes):
        obs = adapter.reset()
        ret, done = 0.0, False
        while not done:
            action = algo.act(obs, deterministic=True)
            obs, r, done, _ = adapter.step(action)
            ret += r
        print("episode %d: return %.1f" % (ep + 1, ret))


if __name__ == "__main__":
    main()
