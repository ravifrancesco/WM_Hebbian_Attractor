import random

import torch
import torch.nn as nn
import torchmetrics.functional as F

import numpy as np

from typing import Callable

from ..utils.utils import activation_pool


def rnn_pool(
    architecture: str,
    input_size: int,
    hidden_size: int,
    num_layers: int,
    nonlinearity: str,
    bias: bool = True,
):
    if architecture.lower() == "rnn":
        model = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity=nonlinearity,
            bias=bias,
            batch_first=True,
        )
    elif architecture.lower() == "lstm":
        model = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=True,
        )
    else:
        raise Exception(f"Model <{architecture}> not implemented")
    model.eval()
    return model


class TileRNN(nn.Module):
    def __init__(
        self,
        architecture: str,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        nonlinearity: str = "tanh",
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.model = rnn_pool(
            architecture, input_size, hidden_size, num_layers, nonlinearity, bias
        )
        self.state = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out, self.state = self.model(x, self.state)
            return out

    def reset_state(self) -> None:
        self.state = None

    def __repr__(self) -> str:
        return self.model.__repr__()


class HashRNN(nn.Module):
    def __init__(
        self,
        architecture: str,
        memsize: int,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        nonlinearity: str = "tanh",
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.model = rnn_pool(
            architecture, input_size, hidden_size, num_layers, nonlinearity, bias
        )
        self.memsize = memsize
        self.out = []
        self.state = []
        self.positions = []

    def forward(self, x: torch.Tensor, pos: int = None) -> torch.Tensor:
        with torch.no_grad():
            if pos in self.positions:
                idx = self.positions.index(pos)
                out, _ = self.model(x, self.state[idx])
            else:
                out, _ = self.model(x)
            return out

    def memorize(self, x: torch.Tensor, pos: int) -> None:
        with torch.no_grad():
            if pos in self.positions:
                idx = self.positions.index(pos)
                self.out[idx], self.state[idx] = self.model(x, self.state[idx])
            elif len(self.out) == self.memsize:  # for now random
                idx = random.randint(0, self.memsize - 1)
                self.out[idx], self.state[idx] = self.model(x, self.state[idx])
                self.positions[idx] = pos
            else:
                o, hc = self.model(x)
                self.out.append(o)
                self.state.append(hc)
                self.positions.append(pos)

    def forget(self, flipped: list[int]) -> None:
        indices = np.where(np.isin(self.positions, flipped))[0]
        self.out = [x for j, x in enumerate(self.out) if j not in indices]
        self.state = [x for j, x in enumerate(self.state) if j not in indices]
        self.positions = [x for j, x in enumerate(self.positions) if j not in indices]

    def reset_state(self) -> None:
        self.out = []
        self.state = []
        self.positions = []

    def get_mem(self) -> torch.Tensor:
        if self.out:
            return torch.stack(self.out), self.positions
        else:
            return None, None

    def __repr__(self) -> str:
        return self.model.__repr__()
    
class Attractor(nn.Module):
    def __init__(
        self,
        dim: int,
        lr: float,
        rr: float,
        decay: float,
        tau: int,
        f: str = "leaky_relu",
    ) -> None:
        super().__init__()

        self.dim = dim

        self.lr = lr
        self.rr = rr
        self.tau = tau
        self.decay = decay

        self.f = activation_pool(f)

        self.reset_state()

    def memorize(self, h: torch.Tensor) -> None:
        self.M = self.rr * self.M + self.lr * (h.T @ h)

    def infer(self, x: torch.Tensor, inhibit: int = None) -> torch.Tensor:
        h = x
        for t in range(self.tau):
            h = self.f(self.decay * h @ self.M)
            if inhibit is not None:
                h[:, inhibit] = 0
        return h

    def reset_state(self) -> None:
        self.M = torch.zeros(self.dim, self.dim)
