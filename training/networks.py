# -*- coding: utf-8 -*-
"""Shared network building blocks for the training algorithms.

Only depends on torch; every builder returns plain nn.Module stacks so the
algorithm modules stay small and readable.
"""

import math

import torch
import torch.nn as nn


def orthogonal_init(module, gain=math.sqrt(2)):
    nn.init.orthogonal_(module.weight, gain)
    if module.bias is not None:
        nn.init.constant_(module.bias, 0.0)
    return module


def mlp(sizes, activation=nn.Tanh, out_activation=None, use_orthogonal=True):
    """Plain MLP; hidden layers use `activation`, the last layer uses
    `out_activation` (identity when None - e.g. critics must stay unbounded)."""
    layers = []
    for i in range(len(sizes) - 1):
        linear = nn.Linear(sizes[i], sizes[i + 1])
        if use_orthogonal:
            gain = math.sqrt(2)
            if out_activation is not None and i == len(sizes) - 2:
                gain = 1.0  # policy/value heads: small init keeps early training stable
            orthogonal_init(linear, gain)
        layers.append(linear)
        last = i == len(sizes) - 2
        if last:
            if out_activation is not None:
                layers.append(out_activation())
        else:
            layers.append(activation())
    return nn.Sequential(*layers)


class NoisyLinear(nn.Module):
    """Factorized Gaussian noisy linear layer (NoisyNet, Fortunato et al. 2018).

    Used by Rainbow instead of epsilon-greedy exploration: the network noise
    itself drives exploration, refreshed before every update.
    """

    def __init__(self, in_features, out_features, sigma_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("weight_epsilon", torch.zeros(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.zeros(out_features))
        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight.data.uniform_(-mu_range, mu_range)
        self.bias.data.uniform_(-mu_range, mu_range)
        init_sigma = self.sigma_init / math.sqrt(self.in_features)
        self.weight_sigma.data.fill_(init_sigma)
        self.bias_sigma.data.fill_(init_sigma)

    def reset_noise(self):
        epsilon_in = torch.randn(self.in_features, device=self.weight.device)
        epsilon_out = torch.randn(self.out_features, device=self.weight.device)
        # factorized gaussian noise: f(x) = sign(x) * sqrt(|x|)
        f_in = torch.sign(epsilon_in) * torch.sqrt(torch.abs(epsilon_in))
        f_out = torch.sign(epsilon_out) * torch.sqrt(torch.abs(epsilon_out))
        self.weight_epsilon.copy_(f_out.outer(f_in))
        self.bias_epsilon.copy_(f_out)

    def forward(self, x):
        if self.training:
            weight = self.weight + self.weight_sigma * self.weight_epsilon
            bias = self.bias + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight
            bias = self.bias
        return nn.functional.linear(x, weight, bias)
