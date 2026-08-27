# -*- coding: utf-8 -*-
"""Algorithm registry: create/load algorithms by name."""

from .ppo import PPO
from .sac import SAC
from .rainbow import Rainbow

ALGORITHMS = {
    "ppo": PPO,
    "sac": SAC,
    "rainbow": Rainbow,
}


def create_algo(name, obs_dim, action_space, cfg=None, device=None):
    if name not in ALGORITHMS:
        raise ValueError("unknown algorithm %r; available: %s" % (name, sorted(ALGORITHMS)))
    return ALGORITHMS[name](obs_dim, action_space, cfg, device)


def load_algo(path, device=None):
    """Load a checkpoint saved by any algorithm's save()."""
    import torch
    payload = torch.load(path, map_location="cpu", weights_only=False)
    name = payload["name"]
    if name not in ALGORITHMS:
        raise ValueError("checkpoint holds unknown algorithm %r" % name)
    return ALGORITHMS[name].load(path, device)
