# -*- coding: utf-8 -*-
"""Unified training entry point for the dogfight environments.

Usage (sandbox must be listening on host:50888 first, e.g.
  cd dogfight_sandbox_hg2/source && ../bin/python/python.exe main.py auto_network):

  python -m training.train --algo ppo  --env oneVSone --timesteps 500000
  python -m training.train --algo sac  --env oneVSone --timesteps 1000000
  python -m training.train --algo rainbow --env oneVSone --timesteps 1000000

Only torch+numpy are required (no new dependencies). One sandbox serves ONE
tcp client, so training uses a single environment instance.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)  # env files live at the repo root

from training.algorithms import ALGORITHMS, create_algo  # noqa: E402
from training.wrapper import DiscreteActionEnv, EnvAdapter  # noqa: E402

ENVS = {}


def register_envs():
    """Lazily import repo-root env classes (they connect on construction)."""
    if ENVS:
        return
    from oneVSoneEnv import oneVSoneEnv
    from twoVStwo import twoVStwo
    from IA_enemy_env import IA_enemy_env
    ENVS.update({
        "oneVSone": oneVSoneEnv,
        "twoVSone": twoVStwo,
        "ia_enemy": IA_enemy_env,
    })


def parse_set(args):
    """--set key=value ... overrides typed hyperparameters."""
    out = {}
    for kv in args:
        key, _, val = kv.partition("=")
        try:
            v = json.loads(val)
        except json.JSONDecodeError:
            v = val
        out[key] = v
    return out


def main():
    ap = argparse.ArgumentParser(description="dogfightEnv modular RL training")
    ap.add_argument("--algo", required=True, choices=sorted(ALGORITHMS))
    ap.add_argument("--env", default="oneVSone", choices=["oneVSone", "twoVSone", "ia_enemy"])
    ap.add_argument("--host", default="192.168.1.103")
    ap.add_argument("--port", default="50888")
    ap.add_argument("--render", action="store_true", help="render the 3D view while training (slower)")
    ap.add_argument("--timesteps", type=int, default=500_000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--save-dir", default=os.path.join(ROOT, "checkpoints"))
    ap.add_argument("--checkpoint-interval", type=int, default=50_000)
    ap.add_argument("--log-interval", type=int, default=10, help="episodes between stdout logs")
    ap.add_argument("--no-normalize", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], help="hyperparameter overrides: --set lr=1e-4 gamma=0.995")
    ap.add_argument("--name", default=None, help="run name (default <algo>_<env>)")
    ap.add_argument("--device", default=None, help="cpu / cuda (default: auto)")
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    register_envs()
    env = ENVS[args.env](host=args.host, port=args.port, rendering=bool(args.render))
    adapter = EnvAdapter(env, normalize=not args.no_normalize)
    if args.algo == "rainbow":
        adapter = DiscreteActionEnv(adapter)

    cfg = parse_set(args.set)
    algo = create_algo(args.algo, adapter.obs_dim, adapter.action_space, cfg, device=args.device)
    run_name = args.name or ("%s_%s" % (args.algo, args.env))
    run_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "log.jsonl")

    print("== training %s on %s | obs_dim=%d | device=%s | %d steps ==" %
          (args.algo, args.env, adapter.obs_dim, algo.device, args.timesteps))
    print("   run dir:", run_dir)

    def save_checkpoint(tag):
        path = os.path.join(run_dir, "model_%s.pt" % tag)
        algo.save(path)
        extra = {"norm": adapter.norm.state_dict() if adapter.norm else None,
                 "env": args.env, "action_grid": getattr(adapter, "actions", None)}
        torch.save(extra, os.path.join(run_dir, "extra_%s.pt" % tag))
        return path

    obs = adapter.reset()
    ep_ret, ep_len, ep_count = 0.0, 0, 0
    recent_returns = []
    best_mean = -float("inf")
    t0 = time.time()
    step = 0
    while step < args.timesteps:
        step += 1

        if algo.name == "ppo":
            action, logp, value = algo.act(obs)
            next_obs, reward, done, _ = adapter.step(action)
            algo.observe(obs, action, logp, value, reward, done)
            stats = {}
            if algo.buffer.ptr >= algo.buffer.size:
                last_value = algo.value_of(next_obs)
                stats = algo.update(last_value)
        elif algo.name in ("sac", "rainbow"):
            action = algo.act(obs)
            next_obs, reward, done, _ = adapter.step(action)
            algo.observe(obs, action, reward, next_obs, done)
            stats = algo.update() if (step % algo.cfg["train_freq"] == 0) else {}
        else:
            raise RuntimeError(algo.name)

        ep_ret += reward
        ep_len += 1
        obs = next_obs

        if done:
            ep_count += 1
            recent_returns.append(ep_ret)
            recent_returns = recent_returns[-100:]
            rec = {"t": step, "episode": ep_count, "ep_ret": round(ep_ret, 2),
                   "ep_len": ep_len, "fps": round(step / max(1e-6, time.time() - t0), 1)}
            rec.update({k: round(v, 5) if isinstance(v, float) else v for k, v in stats.items()})
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            if ep_count % args.log_interval == 0:
                mean100 = float(np.mean(recent_returns))
                print("[%6d] ep %4d  ret %9.1f  mean100 %9.1f  fps %6.1f  %s" %
                      (step, ep_count, ep_ret, mean100, rec["fps"],
                       " ".join("%s=%.4g" % kv for kv in stats.items())))
                if mean100 > best_mean:
                    best_mean = mean100
                    save_checkpoint("best")
            ep_ret, ep_len = 0.0, 0
            obs = adapter.reset()

        if step % args.checkpoint_interval == 0:
            save_checkpoint(str(step))

    final = save_checkpoint("final")
    print("done. final checkpoint:", final)


if __name__ == "__main__":
    main()
