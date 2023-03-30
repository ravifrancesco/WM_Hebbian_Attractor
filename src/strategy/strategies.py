import random

import numpy as np

from typing import Callable

import torch
import torchmetrics.functional as F

from ..memory_game.game import Game
from ..models.cvmodel import CVModel
from ..models.memory import TileRNN


def d_metric_pool(distance_metric: str) -> Callable:
    if distance_metric.lower() == "manhattan":
        return F.pairwise_manhattan_distance
    else:
        raise Exception(f"Distance metric <{distance_metric}> not implemented")


class BaseStrategy:
    def __init__(self, game: Game) -> None:
        self.game = game

    def reset(self) -> None:
        pass

    def reset_turn(self) -> None:
        pass

    def pick(self) -> bool:
        avail = self.game.get_avail()
        pos = random.choice(avail)
        _, _, cont = self.game.pick(pos)
        return cont


class PerfectMemory(BaseStrategy):
    def __init__(self, game: Game, random_pick=False) -> None:
        super().__init__(game)
        self.reset()
        self.random_pick = random_pick

    def reset(self) -> None:
        self.memory = np.full_like(self.game.get_grid_labels(), -1)
        self.curr = None

    def reset_turn(self) -> None:
        self.curr = None

    def pick(self) -> bool:
        avail = self.game.get_avail()
        # Check if the current element match is in memory
        if self.curr in self.memory[avail]:
            indices = np.where(self.memory == self.curr)[0]
            pos = list(set(indices).intersection(set(avail)))[0]
        # Pick randomly (exclude elements which are in memory if possible)
        else:
            indices = avail if self.random_pick else np.where(self.memory < 0)[0]
            choice = list(set(indices).intersection(set(avail)))
            pos = random.choice(choice if choice else avail)
        _, lab, cont = self.game.pick(pos)
        self.curr = lab
        self.memory[pos] = lab
        if not cont:
            self.reset_turn()
        return cont


class TileMemory(BaseStrategy):
    def __init__(
        self,
        game: Game,
        cvmodel: CVModel,
        memory: TileRNN,
        distance_metric: str,
        device: str = "cpu",
    ) -> None:
        super().__init__(game)
        self.device = torch.device(device)
        self.cvmodel = cvmodel
        self.memory = memory
        self.distance_metric = d_metric_pool(distance_metric)
        self.cvmodel.to(self.device)
        self.memory.to(self.device)
        self.reset()

    def reset(self) -> None:
        self.memory.reset_state()
        self.curr = None

    def reset_turn(self) -> None:
        self.curr = None

    def pick(self) -> bool:
        grid_repr = self.__update()
        avail = self.game.get_avail()
        if self.curr is None:
            pos = random.choice(avail)
        else:
            pos = self.__find_closest(self.curr, grid_repr, avail)
        _, _, cont = self.game.pick(pos)
        self.curr = grid_repr[pos]
        if not cont:
            self.reset_turn()
        return cont

    def __update(self) -> torch.Tensor:
        grid = self.game.get_grid().to(self.device)
        cv_out = self.cvmodel(grid)
        cv_out = cv_out.view(cv_out.shape[0], 1, -1)
        return self.memory(cv_out)

    def __find_closest(
        self, x: torch.Tensor, grid_repr: torch.Tensor, avail: list[int]
    ) -> int:
        distances = self.distance_metric(x, grid_repr.flatten(end_dim=1))
        return avail[torch.argmin(distances.flatten()[avail])]
