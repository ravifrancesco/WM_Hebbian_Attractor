import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from typing import Callable



class WorkingMemory(nn.Module):
    def __init__(
        self,
        feature_dim: list[int],
        capacity: int,
        alpha: list[float],
        beta: float,
        eps: float,
        lrfc: float,
        lrcf: float,
        act: Callable = lambda x: torch.clamp(x, min=0, max=1),
        seed: int = None
    ) -> None:
        super().__init__()
        # Set seed
        if seed is not None:
            torch.manual_seed(seed)
        # Set feature unit weights
        self.feature_dim = feature_dim
        self.feature_units = sum(self.feature_dim)
        self.Wff = nn.Linear(self.feature_units, self.feature_units, bias=False)
        # Set conjunctive unit weights
        self.capacity = capacity
        self.Wcc = nn.Linear(self.capacity, self.capacity, bias=False)
        # Set feature-conjunctive unit weights
        self.Wfc = nn.Linear(self.feature_units, self.capacity, bias=False)
        self.Wcf = nn.Linear(self.capacity, self.feature_units, bias=False)
        # Set hyperparameters
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.lrfc = lrfc
        self.lrcf = lrcf
        self.act = act
        # Initialize weights
        self.init_W()
        # Initialize memory
        self.c = torch.zeros(self.capacity)
        self.f = torch.zeros(self.feature_units)
        # Reset history
        self.reset()

    def init_W(self) -> None:
        # Initialize Wff
        bd = torch.block_diag(*[torch.ones(s, s) for s in self.feature_dim])
        # bd[:9, :9] *= 0.7
        # bd[np.diag_indices(9)] = 1
        eye = torch.eye(self.feature_units)
        self.Wff.weight.data = self.alpha[4] * eye + self.alpha[3] * bd
        # self.Wff.weight.data = torch.rand(self.feature_units, self.feature_units)
        # Initialize Wcc
        eye = torch.eye(self.capacity)
        ones = torch.ones(self.capacity)
        self.Wcc.weight.data = self.alpha[1] * eye + self.alpha[0] * ones
        # Initialize Wfc Wcf
        wfc = torch.rand(self.capacity, self.feature_units)
        wcf = torch.rand(self.feature_units, self.capacity)
        self.Wfc.weight.data = wfc
        self.Wcf.weight.data = wcf if self.lrcf != self.lrfc else wfc.T

    def reset(self) -> None:
        self.c_history = []
        self.f_history = []
        self.stimulus_history = []

    def forward(self, stimulus: torch.tensor, learn: bool = True) -> torch.Tensor:
        with torch.no_grad():
            self.f = self.act(
                self.beta
                + self.Wff(self.f - self.beta)
                + self.alpha[2] * self.Wcf(self.c - self.beta)
                + stimulus
            )

            self.c = self.act(
                self.beta
                + self.Wcc(self.c - self.beta)
                + self.alpha[5] * self.Wfc(self.f - self.beta)
                + self.eps * torch.randn(self.capacity)
            )

            # deltaW = torch.outer(self.f - self.beta, self.f - self.beta)
            # self.Wff.weight.data = torch.clamp(self.Wff.weight.data + self.lrfc * deltaW.T, min=-1, max=1)
            # deltaW = torch.outer(self.c - self.beta, self.c - self.beta)
            # self.Wcc.weight.data = torch.clamp(self.Wcc.weight.data + self.lrfc * deltaW.T, min=0, max=1)
            if learn:
                deltaW = torch.outer(self.f - self.beta, self.c - self.beta)
                self.Wfc.weight.data = self.act(
                    self.Wfc.weight.data + self.lrfc * deltaW.T
                )
                self.Wcf.weight.data = self.act( 
                    self.Wcf.weight.data + self.lrcf * deltaW
                )

            self.stimulus_history.append(stimulus.detach())
            self.f_history.append(self.f.detach())
            self.c_history.append(self.c.detach())

            return self.f.detach()

    def get_activation_history(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return torch.vstack(self.f_history), torch.vstack(self.c_history), torch.vstack(self.stimulus_history)
