import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import math

import numpy as np

from typing import Callable


def activation_pool(activation: str) -> Callable:
    """Returns the selected activation function.

    Args:
        activation (str): Name of the activation function.

    Raises:
        Exception: The selected activation function is not implemented.

    Returns:
        Callable: Activation function.
    """
    if activation.lower() == "leaky_relu":
        return torch.nn.functional.leaky_relu
    elif activation.lower() == "relu":
        return torch.nn.functional.relu
    elif activation.lower() == "tanh":
        return torch.nn.functional.tanh
    elif activation.lower() == "sigmoid":
        return torch.nn.functional.sigmoid
    else:
        raise Exception(f"Activation <{activation}> not implemented")


class FastAttractor(nn.Module):
    """Hebbian attractor network memory."""

    def __init__(
        self,
        dim: int,
        lr: float,
        rr: float,
        f: str = "leaky_relu",
    ) -> None:
        """_summary_

        Args:
            dim (int): Total number of tiles in the grid.
            lr (float): Learning rate.
            rr (float): Memory decay.
            f (str, optional): Activation function. Defaults to "leaky_relu".
        """
        super().__init__()

        self.dim = dim

        self.lr = nn.Parameter(torch.tensor(lr))
        self.rr = nn.Parameter(torch.tensor(rr))
        self.f = activation_pool(f)

        self.register_buffer("M", torch.rand(dim, dim))
        self.register_buffer("hs", torch.zeros(1, dim))

        self.reset_state()

    def forward(self, x: torch.Tensor, steps: int) -> torch.Tensor:
        """Forward function. Used for both memorization and inference.

        Args:
            x (torch.Tensor): Input to the network.
            steps (int): Number of timesteps (updates).

        Returns:
            torch.Tensor: The current network state.
        """
        for i in range(steps):
            self.hs = self.f(F.normalize(x + self.hs @ self.M))
            self.M = self.rr * self.M + self.lr * (self.hs.T @ self.hs)
            self.history.append(self.hs.detach())
            self.e_history.append(-self.hs @ self.M @ self.hs.T)

        return self.hs.detach()

    def get_activation_history(self) -> torch.Tensor:
        """Returns the history of the network activations.

        Returns:
            torch.Tensor: Tensor representing the history of activations.
        """
        return torch.vstack(self.history)

    def get_energy_history(self) -> torch.Tensor:
        """Returns the history of the network energy.

        Returns:
            torch.Tensor: Tensor representing the history of network energy.
        """
        return torch.tensor(self.e_history)

    def reset_state(self) -> None:
        """Resets the network memory, state and histories."""
        self.M[True] = 0
        self.hs[True] = 0
        self.history = []
        self.e_history = []
