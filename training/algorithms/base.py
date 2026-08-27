# -*- coding: utf-8 -*-
"""Algorithm base class: device handling, config, serialization."""

import os

import torch


def default_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


class BaseAlgorithm:
    """Common contract for all algorithms.

    act(obs, deterministic)  -> action in the env's raw action space
    update(...)              -> per-algo signature, called by the trainer
    save(path) / load(path)  -> checkpoint a single .pt file
    """

    name = "base"

    def __init__(self, obs_dim, action_space, cfg=None, device=None):
        self.obs_dim = int(obs_dim)
        self.action_space = action_space
        self.cfg = dict(self.DEFAULTS)
        if cfg:
            self.cfg.update(cfg)
        self.device = torch.device(device) if device else torch.device(default_device())

    DEFAULTS = {}

    # ------------------------------------------------------------------ helpers

    def cfg_get(self, key, fallback=None):
        return self.cfg.get(key, self.DEFAULTS.get(key, fallback))

    def save(self, path):
        payload = {
            "name": self.name,
            "cfg": self.cfg,
            "obs_dim": self.obs_dim,
        }
        return payload

    def load_state(self, payload):
        pass

    @staticmethod
    def _ensure_dir(path):
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
