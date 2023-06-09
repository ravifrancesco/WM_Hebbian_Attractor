import random

import torch
import torch.nn as nn
import torch.nn.functional as F 

import numpy as np

from typing import Callable


def activation_pool(activation: str) -> Callable:
    if activation.lower() == "leaky_relu":
        return torch.nn.functional.leaky_relu
    elif activation.lower() == "tanh":
        return torch.nn.functional.tanh
    else:
        raise Exception(f"Activation <{activation}> not implemented")


class FastAttractor(nn.Module):
    def __init__(
        self,
        dim: int,
        lr: float,
        rr: float,
        f: str = "leaky_relu",
    ) -> None:
        super().__init__()

        self.dim = dim

        self.lr = nn.Parameter(torch.tensor(lr))
        self.rr = nn.Parameter(torch.tensor(rr))
        self.f = activation_pool(f)

        self.register_buffer('M', torch.rand(dim, dim))
        self.register_buffer('hs', torch.zeros(1, dim))

        self.history = []
        self.e_history = []

    def forward(self, x: torch.Tensor, steps: int):

        for i in range(steps):
            self.hs = self.f(F.normalize(x + self.hs @ self.M))
            self.M = self.rr * self.M + self.lr * (self.hs.T @ self.hs)
            self.history.append(self.hs.detach())
            self.e_history.append(self.hs @ self.M @ self.hs.T)

        return self.hs.detach()

    def get_activation_history(self) -> torch.Tensor:
        return torch.vstack(self.history)

    def get_energy_history(self) -> torch.Tensor:
        return torch.tensor(self.e_history)

    def reset_state(self) -> None:
        self.M[True] = 0
