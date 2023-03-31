import torch
import torch.nn as nn
import torchmetrics.functional as F

import numpy as np

from typing import Callable


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out, _ = self.model(x)
            return out

    def memorize(self, x: torch.Tensor, pos: int, idx: int) -> None:
        if idx >= self.memsize:
            raise Exception(f"Index <{idx}> exceeds memory size <{self.memsize}>")
        with torch.no_grad():
            if len(self.out) > idx:
                self.out[idx], self.state[idx] = self.model(x, self.state[idx])
                self.locations[idx] = pos
            else:
                o, hc = self.model(x)
                self.out.append(o)
                self.state.append(hc)
                self.positions.append(pos)

    def reset_state(self) -> None:
        self.out = []
        self.state = []
        self.positions = []

    def get_out(self, avail: list[int]) -> None:
        avail_o = np.where(np.isin(self.positions, avail))[0]
        return torch.stack(self.out[avail_o])

    def __repr__(self) -> str:
        return self.model.__repr__()
