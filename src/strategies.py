import random

import numpy as np

from typing import Callable

import torch
import torch.nn.functional as F

from game import Game
from memory import FastAttractor


class BaseStrategy:
    def __init__(self, game: Game) -> None:
        self.game = game

    def reset(self) -> None:
        pass

    def reset_turn(self) -> None:
        pass

    def pick(self) -> int:
        avail = np.where(self.game.get_avail())[0]
        pos = np.random.choice(avail)
        _, _, cont = self.game.pick(pos)
        return pos


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

    def pick(self) -> int:
        avail = np.where(self.game.get_avail())[0]
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
        return pos


class BernoulliMemory(BaseStrategy):
    def __init__(
        self, game: Game, p: float = 1.0, tau: float = 0.1, random_pick=False
    ) -> None:
        super().__init__(game)
        self.reset()
        self.p = p
        self.tau = tau
        self.random_pick = random_pick

    def reset(self) -> None:
        self.memory = np.full_like(self.game.get_grid_labels(), -1)
        self.decay = np.zeros_like(self.game.get_grid_labels(), dtype=np.float64)
        self.curr = None

    def reset_turn(self) -> None:
        self.curr = None

    def pick(self) -> int:
        avail = np.where(self.game.get_avail())[0]
        t_memory = self.memory
        # Memory decay
        self.decay *= self.tau
        forget = np.random.rand(*self.memory.shape) > self.decay
        t_memory[forget] = -1
        # Check if the current element match is in memory
        if self.curr in t_memory[avail]:
            indices = np.where(t_memory == self.curr)[0]
            pos = list(set(indices).intersection(set(avail)))[0]
        # Pick randomly (exclude elements which are in memory if possible)
        else:
            indices = avail if self.random_pick else np.where(t_memory < 0)[0]
            choice = list(set(indices).intersection(set(avail)))
            pos = random.choice(choice if choice else avail)
        _, lab, cont = self.game.pick(pos)
        self.curr = lab
        if random.random() <= self.p:
            self.memory[pos] = lab
            self.decay[pos] = 1
        if not cont:
            self.reset_turn()
        return pos

class FastAttractorMemory(BaseStrategy):
    def __init__(
        self,
        game: Game,
        memory: FastAttractor,
        dim: int,
        steps: 20,
        familiar: bool = False,
        device: str = "cpu",
    ) -> None:
        super().__init__(game)
        self.device = torch.device(device)
        self.memory = memory
        self.memory.to(self.device)

        self.nue = game.get_n_labels()
        self.dim = dim
        self.steps = steps

        self.familiar = familiar
        self.seen = np.full(np.prod(game.get_board_size()), -1)

        self.reset()

    def reset(self) -> None:
        self.curr = None
        self.memory.reset_state()

    def reset_turn(self) -> None:
        self.curr = None

    # TODO try to set to -1 the ones that are not accessible
    def pick(self) -> int:
        avail = self.game.get_avail()
        avail_idx = np.where(avail)[0]
        if self.curr is None or (self.familiar and self.curr not in self.seen[avail]):
            pos = random.choice(avail_idx)
        else:
            unavail = np.where(~avail)[0]
            pos_oh = torch.zeros(self.dim)
            curr_oh = torch.zeros(self.nue)
            curr_oh[self.curr] = 1.
            pos_oh[unavail] = -1.
            x = torch.unsqueeze(torch.concat([pos_oh, curr_oh]), dim=0)
            positions = self.memory(x.to(self.device), self.steps)[: self.dim]
            pos = avail_idx[torch.argmax(positions.flatten()[avail_idx])]
        _, self.curr, cont = self.game.pick(pos)
        self.seen[pos] = self.curr
        # curr_oh = F.one_hot(torch.tensor(self.curr), self.nue)
        # pos_oh = F.one_hot(torch.tensor(pos), self.dim)
        curr_oh = torch.full([self.nue], -1.)
        curr_oh[self.curr] = 1.
        pos_oh = torch.full([self.dim], -1.)
        pos_oh[pos] = 1.
        h = torch.unsqueeze(torch.concat([pos_oh, curr_oh]), dim=0)
        self.memory(h.to(self.device), self.steps)
        if not cont:
            self.reset_turn()
        return pos