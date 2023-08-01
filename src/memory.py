import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import math

import numpy as np

from typing import Callable


def activation_pool(activation: str) -> Callable:
    if activation.lower() == "leaky_relu":
        return torch.nn.functional.leaky_relu
    elif activation.lower() == "relu":
        return torch.nn.functional.relu
    elif activation.lower() == "tanh":
        return torch.nn.functional.tanh
    elif activation.lower() == 'sigmoid':
        return torch.nn.functional.sigmoid
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

        self.register_buffer("M", torch.rand(dim, dim))
        self.register_buffer("hs", torch.zeros(1, dim))

        self.history = []
        self.e_history = []

    def forward(self, x: torch.Tensor, steps: int):

        for i in range(steps):
            self.hs = self.f(F.normalize(x + self.hs @ self.M))
            self.M = self.rr * self.M + self.lr * (self.hs.T @ self.hs)
            self.history.append(self.hs.detach())
            self.e_history.append(-self.hs @ self.M @ self.hs.T)

        return self.hs.detach()

    def get_activation_history(self) -> torch.Tensor:
        return torch.vstack(self.history)

    def get_energy_history(self) -> torch.Tensor:
        return torch.tensor(self.e_history)

    def reset_state(self) -> None:
        self.M[True] = 0


class PottsAttractor(nn.Module):
    def __init__(
        self,
        H: int,
        M: int,
        ga: float,
        gw: float,
        gwr: float,
        gb: float,
        gs: float,
        k: float,
        dt: float = 1e-3,
        tm: float = 5e-2,
        ta: float = 2.7,
        tzi: float = 2.4e-1,
        tzj: float = 2.4e-1,
        tp: float = 10.0,
    ) -> None:
        super().__init__()

        self.H = H
        self.M = M

        self.ga = nn.Parameter(torch.tensor(ga))
        self.gw = nn.Parameter(torch.tensor(gw))
        self.gwr = nn.Parameter(torch.tensor(gwr))
        self.gb = nn.Parameter(torch.tensor(gb))
        self.gs = nn.Parameter(torch.tensor(gs))

        self.k = nn.Parameter(torch.tensor(k))

        self.dt = dt
        self.tm = tm
        self.ta = ta
        self.tzi = tzi
        self.tzj = tzj
        self.tp = tp

        self.eps = torch.finfo(torch.float32).tiny

        self.reset_state()

    def reset_state(self) -> None:
        v = 1 / self.M

        self.register_buffer("a", torch.zeros(self.H, self.M))
        self.register_buffer("s", torch.full((self.H, self.M), math.log(v)))
        self.register_buffer("o", torch.full((self.H, self.M), v))

        self.register_buffer("zi", torch.full((self.H, self.M), v))
        self.register_buffer("zj", torch.full((self.H, self.M), v))

        self.register_buffer("p", torch.tensor(0.0))
        self.register_buffer("pi", torch.full((self.H, self.M), v))
        self.register_buffer("pj", torch.full((self.H, self.M), v))
        self.register_buffer("pij", torch.full((self.H * self.M, self.H * self.M), 1 / (self.M**2)))

        self.register_buffer("b", torch.full((self.H, self.M), self.gw.item() * math.log(v)))

        self.register_buffer("w", torch.zeros(self.H * self.M, self.H * self.M))

    def log_eps(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.where(x <= 0.0, self.eps, x))

    def update_state(self, x: torch.Tensor, gw: torch.Tensor) -> None:
        st = gw * (self.b.flatten() + self.w.T @ self.o.flatten()).flatten()
        self.s += (self.dt / self.tm) * (
            st.view_as(self.s)
            - self.a
            + self.log_eps(x)
            + self.gs * torch.randn_like(self.s)
            - self.s
        )
        self.o = F.softmax(self.s, dim=1)
        self.a += (self.dt / self.ta) * (self.ga * self.o - self.a)

    def update_traces(self) -> None:
        self.zi += (self.dt / self.tzi) * (self.o - self.zi)
        self.zj += (self.dt / self.tzj) * (self.o - self.zj)

    def update_probabilities(self) -> None:
        self.p += (self.k * self.dt / self.tp) * (1 - self.p)
        self.pi += (self.k * self.dt / self.tp) * (self.zi - self.pi)
        self.pj += (self.k * self.dt / self.tp) * (self.zj - self.pj)
        self.pij += (self.k * self.dt / self.tp) * (torch.outer(self.zi.flatten(), self.zj.flatten()) - self.pij)

    def update_connections(self) -> None:
        self.w = self.log_eps((self.p * self.pij) / torch.outer(self.pi.flatten(), self.pj.flatten()))
        self.b = self.gb * self.log_eps(self.pj)

    def forward(self, x: torch.Tensor, train=True) -> torch.Tensor:
        with torch.no_grad():
            self.update_state(x, self.gw if train else self.gwr)
            self.update_traces()
            if train:
                self.update_probabilities()
                self.update_connections()
        return self.o.detach()
