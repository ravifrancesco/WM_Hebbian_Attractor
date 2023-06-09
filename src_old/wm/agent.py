import random

import numpy as np

from typing import Callable

import torch
import torch.nn.functional as F

from .wm import WorkingMemory
from ..memory_game.game import Game


class Agent:
    def __init__(
        self,
        game: Game,
        wm: WorkingMemory,
        reset_steps: int = 200,
        display_steps: int = 120,
        delay_steps: int = 50,
        response_steps: int=220,
        seed: int = None
    ) -> None:
        if seed is not None:
            random.seed(seed)
        self.game = game
        self.wm = wm

        self.reset_steps = reset_steps
        self.display_steps = display_steps
        self.delay_steps = delay_steps
        self.response_steps = response_steps

        self.nue = len(np.unique(game.get_grid_labels()))
        self.grid_dim = np.prod(game.get_grid_labels().shape)
        # self.f_size = self.nue + self.dim
        self.dim = self.wm.feature_dim[0]
        self.f_size = self.wm.feature_units


        self.reset()

    def reset(self) -> None:
        self.t_steps = [self.reset_steps]
        self.events = []
        self.curr = None
        self.show(torch.full([self.f_size], -1.0), self.reset_steps)

    def reset_turn(self) -> None:
        self.curr = None

    def pick(self) -> int: # TODO make the t_steps better
        avail = self.game.get_avail()
        if self.curr is None:
            pos = random.choice(avail)
        else:
            unavail = self.game.get_avail(unavailable=True)
            p_stimulus = torch.full([self.f_size], -1.0)
            p_stimulus[self.dim + self.curr] = 1.0
            self.show(p_stimulus, self.display_steps, learn=True)
            self.t_steps.append(self.t_steps[-1] + self.display_steps)
            self.events.append('probe')
            r_stimulus = torch.zeros(self.f_size)
            r_stimulus[unavail] = -1.0
            r_stimulus[self.grid_dim:self.dim] = -1.0
            r_stimulus[self.dim+self.nue:] = -1.0
            f = self.show(r_stimulus, self.response_steps, learn=True)
            self.t_steps.append(self.t_steps[-1] + self.response_steps)
            self.events.append('response')
            # print(f'retrieved: {f}')
            d_stimulus = torch.zeros(self.f_size)
            #d_stimulus[unavail] = -1.0
            self.show(d_stimulus, self.delay_steps, learn=True)
            self.t_steps.append(self.t_steps[-1] + self.delay_steps)
            self.events.append('delay')
            # print(f'pick from {self.game.get_avail()}')
            pos = avail[torch.argmax(f[: self.dim][avail])]
        _, self.curr, cont = self.game.pick(pos)
        d_stimulus = torch.full([self.f_size], -1.0)
        d_stimulus[[pos, self.dim + self.curr]] = 1.0
        f = self.show(d_stimulus, self.display_steps)
        self.t_steps.append(self.t_steps[-1] + self.display_steps)
        self.events.append('stimulus')
        self.show(torch.zeros(self.f_size), self.delay_steps)
        self.t_steps.append(self.t_steps[-1] + self.delay_steps)
        self.events.append('delay')
        if not cont:
            self.reset_turn()
        return pos

    def show(
        self, stimulus: torch.Tensor, steps: int, learn: bool = True
    ) -> torch.Tensor:
        # print(f'{steps}: {stimulus[:9]} - {stimulus[9:]}')
        for i in range(steps):
            f = self.wm(stimulus, learn=learn)
        return f
