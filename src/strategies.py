import random

import numpy as np

from typing import Callable

import torch
import torch.nn.functional as F


from game import Game
from memory import FastAttractor, PottsAttractor


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
        steps: 20,
        device: str = "cpu",
    ) -> None:
        super().__init__(game)
        self.device = torch.device(device)
        self.memory = memory
        self.memory.to(self.device)

        self.nue = game.get_n_labels()
        self.dim = np.prod(game.get_board_size())
        self.steps = steps

        self.seen = np.full(np.prod(game.get_board_size()), -1)

        self.reset()

    def reset(self) -> None:
        self.curr = None
        self.memory.reset_state()

    def reset_turn(self) -> None:
        self.curr = None

    def pick(self) -> int:
        avail = self.game.get_avail()
        avail_idx = np.where(avail)[0]
        if self.curr is None:
            pos = random.choice(avail_idx)
        else:
            unavail = np.where(~avail)[0]
            pos_oh = torch.zeros(self.dim)
            curr_oh = torch.zeros(self.nue)
            curr_oh[self.curr] = 1.0 # TODO try with zeros
            pos_oh[unavail] = -1.0
            x = torch.unsqueeze(torch.concat([pos_oh, curr_oh]), dim=0)
            positions = self.memory(x.to(self.device), self.steps)[: self.dim]
            p = positions.flatten()
            max_v = torch.max(p[avail_idx])
            pos = np.atleast_1d(avail_idx[torch.where(p[avail_idx] == max_v)[0]])
            pos = random.choice(pos)
        _, self.curr, cont = self.game.pick(pos)
        curr_oh = torch.full([self.nue], -1.0)
        curr_oh[self.curr] = 1.0
        pos_oh = torch.full([self.dim], -1.0)
        pos_oh[pos] = 1.0
        h = torch.unsqueeze(torch.concat([pos_oh, curr_oh]), dim=0)
        self.memory(h.to(self.device), self.steps)
        if not cont:
            self.reset_turn()
            # self.memory(torch.zeros(self.dim + self.nue).to(self.device), self.steps)
        return pos


class PottsAttractorMemory(BaseStrategy):
    def __init__(
        self,
        game: Game,
        memory: PottsAttractor,
        threshold: float = 11.0,
        memorization: int = 1000,
        reaction: int = 2500,
        ratio: int = 1,
        device="cpu",
    ) -> None:
        super().__init__(game)

        self.device = device

        self.memory = memory.to(self.device)
        self.threshold = threshold

        self.memorization = memorization
        self.reaction = reaction

        self.nue = game.get_n_labels()
        self.dim = np.prod(game.get_board_size())

        div = ratio + 1
        self.pos_H = (self.memory.H // div) * ratio
        self.lab_H = self.memory.H - self.pos_H

        self.reset()

    def fill_patterns(self, H: int, n_patterns: int) -> torch.Tensor: # TODO check if best option
        return torch.stack([
            torch.eye(self.memory.M).roll(shifts=i, dims=1)[:H]
            for i in range(n_patterns)
        ], dim=0).to(self.device)

    def set_patterns(self) -> None:
        self.pos_patterns = self.fill_patterns(self.pos_H, self.dim)
        self.lab_patterns = self.fill_patterns(self.lab_H, self.nue)
        self.infer_pos = torch.ones(self.pos_H, self.memory.M).to(self.device)

    def reset(self) -> None:
        self.curr = None
        self.set_patterns()
        self.memory.reset_state()

    def reset_turn(self) -> None:
        self.curr = None

    def pick(self) -> int:
        avail = self.game.get_avail()
        avail_idx = np.where(avail)[0]
        if self.curr is None:
            pos = random.choice(avail_idx)
        else:
            unavail = np.where(~avail)[0]
            m = torch.zeros(self.dim)
            # pattern = torch.vstack((torch.clamp(self.infer_pos - torch.sum(self.pos_patterns[unavail], dim=0), 0, 1), self.lab_patterns[self.curr]))
            pattern = torch.vstack((self.infer_pos, self.lab_patterns[self.curr]))
            for i in range(self.reaction):
                o = self.memory(pattern, train=False)[:self.pos_H]
                m += F.cosine_similarity(o.flatten(), self.pos_patterns.flatten(1,2))
                if torch.any(m > self.threshold):
                    break
            pos = avail_idx[torch.argmax(m[avail_idx])]
        _, self.curr, cont = self.game.pick(pos)
        pattern = torch.vstack((self.pos_patterns[pos], self.lab_patterns[self.curr]))
        for i in range(self.memorization):
            self.memory(pattern)
        if not cont:
            self.reset_turn()
        return pos