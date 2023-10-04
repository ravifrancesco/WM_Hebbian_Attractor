import random

import numpy as np

from typing import Callable

import torch
import torch.nn.functional as F


from game import Game
from memory import FastAttractor


class BaseStrategy:
    """Random strategy for playing the memory matching game."""

    def __init__(self, game: Game) -> None:
        """Initialization.

        Args:
            game (Game): Game to play.
        """
        self.game = game

    def reset(self) -> None:
        """Resets the strategy."""
        pass

    def reset_turn(self) -> None:
        """Resets the strategy' for this trial."""
        pass

    def pick(self) -> int:
        """Function used to pick a tile. Both the first and second tiles are picked uniformly at random.

        Returns:
            int: Position of the selected tile.
        """
        avail = np.where(self.game.get_avail())[0]
        pos = np.random.choice(avail)
        _, _, cont = self.game.pick(pos)
        return pos


class PerfectMemory(BaseStrategy):
    """Perfect memory strategy."""

    def __init__(self, game: Game, random_pick=False) -> None:
        """Initialization.

        Args:
            game (Game): Game to play.
            random_pick (bool, optional): If True, the first tile is picked at random, if False, if all the tiles belonging to one label have already been seen, on of these tiles is selected. Defaults to False.
        """
        super().__init__(game)
        self.reset()
        self.random_pick = random_pick

    def reset(self) -> None:
        """Resets the strategy."""
        self.memory = np.full_like(self.game.get_grid_labels(), -1)
        self.curr = None

    def reset_turn(self) -> None:
        """Resets the strategy' for this trial."""
        self.curr = None

    def pick(self) -> int:
        """Function used to pick a tile. The first tile is picked according to the value of 'random_pick'. The second tile is picked randomly if the first tile's label has not been seen before, otherwise the correct matching tile is selected.

        Returns:
            int: Position of the selected tile.
        """
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


class FastAttractorMemory(BaseStrategy):
    """Attractor network memory strategy."""

    def __init__(
        self,
        game: Game,
        memory: FastAttractor,
        steps: 10,
        device: str = "cpu",
    ) -> None:
        """Initialization.

        Args:
            game (Game): Game to play.
            memory (FastAttractor): Attractor memory network.
            steps (20): Number of steps for the attractor memory network.
            device (str, optional): Device for the network. Defaults to "cpu".
        """
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
        """Resets the strategy."""
        self.curr = None
        self.memory.reset_state()

    def reset_turn(self) -> None:
        """Resets the strategy' for this trial."""
        self.curr = None

    def pick(self) -> int:
        """Function used to pick a tile. The first tile is picked at random. The second tile is picked using the most active position unit of the network.

        Returns:
            int:  Position of the selected tile.
        """
        avail = self.game.get_avail()
        avail_idx = np.where(avail)[0]
        if self.curr is None:
            pos = random.choice(avail_idx)
        else:
            unavail = np.where(~avail)[0]
            pos_oh = torch.zeros(self.dim)
            curr_oh = torch.zeros(self.nue)
            curr_oh[self.curr] = 1.0
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
        return pos
