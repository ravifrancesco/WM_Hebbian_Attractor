import random

import numpy as np

from typing import Callable

import torch
import torch.nn.functional as F

from ..memory_game.game import Game
from ..models.cvmodel import CVModel
from ..models.memory import TileRNN, HashRNN, Attractor, FastAttractor
from ..utils.utils import d_metric_pool


class BaseStrategy:
    def __init__(self, game: Game) -> None:
        self.game = game

    def reset(self) -> None:
        pass

    def reset_turn(self) -> None:
        pass

    def pick(self) -> int:
        avail = self.game.get_avail()
        pos = random.choice(avail)
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
        avail = self.game.get_avail()
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
        self.distance_metric = distance_metric
        self.cvmodel.to(self.device)
        self.memory.to(self.device)
        self.reset()

    def reset(self) -> None:
        self.memory.reset_state()
        self.curr = None

    def reset_turn(self) -> None:
        self.curr = None

    def pick(self) -> int:
        grid_repr = self.__update()
        avail = self.game.get_avail()
        if self.curr is None:
            pos = random.choice(avail)
        else:
            pos = self.__find(self.curr, grid_repr, avail)
        _, _, cont = self.game.pick(pos)
        self.curr = grid_repr[pos]
        if not cont:
            self.reset_turn()
        return pos

    def __update(self) -> torch.Tensor:
        grid = self.game.get_grid().to(self.device)
        cv_out = self.cvmodel(grid)
        cv_out = cv_out.view(cv_out.shape[0], 1, -1)
        return self.memory(cv_out)

    def __find(self, x: torch.Tensor, grid_repr: torch.Tensor, avail: list[int]) -> int:
        distances = d_metric_pool(self.distance_metric, x, grid_repr.flatten(end_dim=1))
        # print(distances)
        return avail[torch.argmin(distances.flatten()[avail])]


class RandomHashMemory(BaseStrategy):
    def __init__(
        self,
        game: Game,
        cvmodel: CVModel,
        memory: HashRNN,
        distance_metric: str,
        device: str = "cpu",
    ) -> None:
        super().__init__(game)
        self.device = torch.device(device)
        self.cvmodel = cvmodel
        self.memory = memory
        self.distance_metric = distance_metric
        self.cvmodel.to(self.device)
        self.memory.to(self.device)
        self.reset()

    def reset(self) -> None:
        self.memory.reset_state()
        self.curr = None
        self.avrg_dist = 0

    def reset_turn(self) -> None:
        self.curr = None

    def pick(self) -> int:
        avail = self.game.get_avail()
        mem, positions = self.memory.get_mem()
        avail_o = np.where(np.isin(positions, avail))[0]
        if self.curr is None or mem is None or not len(avail_o):
            pos = random.choice(avail)
        else:
            pos = self.__find(self.curr, mem, positions, avail_o)
        img, _, cont = self.game.pick(pos)
        self.curr, cv_out = self.__get_repr(img.unsqueeze(0), pos)
        self.memory.memorize(cv_out, pos)
        self.memory.forget(self.game.get_flipped())
        if not cont:
            self.reset_turn()
        return pos

    def __get_repr(self, x: torch.Tensor, pos: int) -> torch.Tensor:
        x = x.to(self.device)
        cv_out = self.cvmodel(x)
        return self.memory(cv_out, pos), cv_out

    def __find(
        self,
        x: torch.Tensor,
        grid_repr: torch.Tensor,
        positions: list[int],
        avail_o: list[int],
    ) -> int:
        distances = d_metric_pool(self.distance_metric, x, grid_repr.flatten(end_dim=1))
        return min((distances[i], positions[i]) for i in avail_o)[1]


class BaseAttractorMemory(BaseStrategy):
    def __init__(
        self,
        game: Game,
        memory: Attractor,
        dim: int,
        familiar: bool = False,
        device: str = "cpu",
    ) -> None:
        super().__init__(game)
        self.device = torch.device(device)
        self.memory = memory
        self.memory.to(self.device)

        self.nue = len(np.unique(game.get_grid_labels()))
        self.dim = dim

        self.familiar = familiar
        self.seen = np.array([-1] * len(game.get_grid_labels()))

        self.reset()

    def reset(self) -> None:
        self.curr = None
        self.memory.reset_state()

    def reset_turn(self) -> None:
        self.curr = None

    # TODO try to set to -1 the ones that are not accessible
    def pick(self) -> int:
        avail = self.game.get_avail()
        if self.curr is None or (self.familiar and self.curr not in self.seen[avail]):
            pos = random.choice(avail)
        else:
            unavail = self.game.get_avail(unavailable=True)
            pos_oh = torch.zeros(self.dim)
            curr_oh = torch.zeros(self.nue)
            curr_oh[self.curr] = 1.
            x = torch.unsqueeze(torch.concat([pos_oh, curr_oh]), dim=0)
            positions = self.memory.infer(x, unavail)[: self.dim]
            pos = avail[torch.argmax(positions.flatten()[avail])]
        _, self.curr, cont = self.game.pick(pos)
        self.seen[pos] = self.curr
        # curr_oh = F.one_hot(torch.tensor(self.curr), self.nue)
        # pos_oh = F.one_hot(torch.tensor(pos), self.dim)
        curr_oh = torch.full([self.nue], -1.)
        curr_oh[self.curr] = 1.
        pos_oh = torch.full([self.dim], -1.)
        pos_oh[pos] = 1.
        h = torch.unsqueeze(torch.concat([pos_oh, curr_oh]), dim=0)
        self.memory.memorize(h)
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

        self.nue = len(np.unique(game.get_grid_labels()))
        self.dim = dim
        self.steps = steps

        self.familiar = familiar
        self.seen = np.array([-1] * len(game.get_grid_labels()))

        self.reset()

    def reset(self) -> None:
        self.curr = None
        self.memory.reset_state()

    def reset_turn(self) -> None:
        self.curr = None

    # TODO try to set to -1 the ones that are not accessible
    def pick(self) -> int:
        avail = self.game.get_avail()
        if self.curr is None or (self.familiar and self.curr not in self.seen[avail]):
            pos = random.choice(avail)
        else:
            unavail = self.game.get_avail(unavailable=True)
            pos_oh = torch.zeros(self.dim)
            curr_oh = torch.zeros(self.nue)
            curr_oh[self.curr] = 1.
            pos_oh[unavail] = -1.
            x = torch.unsqueeze(torch.concat([pos_oh, curr_oh]), dim=0)
            positions = self.memory(x, self.steps)[: self.dim]
            pos = avail[torch.argmax(positions.flatten()[avail])]
        _, self.curr, cont = self.game.pick(pos)
        self.seen[pos] = self.curr
        # curr_oh = F.one_hot(torch.tensor(self.curr), self.nue)
        # pos_oh = F.one_hot(torch.tensor(pos), self.dim)
        curr_oh = torch.full([self.nue], -1.)
        curr_oh[self.curr] = 1.
        pos_oh = torch.full([self.dim], -1.)
        pos_oh[pos] = 1.
        h = torch.unsqueeze(torch.concat([pos_oh, curr_oh]), dim=0)
        self.memory(h, self.steps)
        if not cont:
            self.reset_turn()
        return pos

class RandomAttractorMemory(BaseStrategy):
    def __init__(
        self,
        game: Game,
        memory: Attractor,
        capacity: int,
        device: str = "cpu",
    ) -> None:
        super().__init__(game)
        self.device = torch.device(device)
        self.memory = memory
        self.memory.to(self.device)

        self.dim = len(self.game.get_grid_labels())
        self.capacity = capacity

        self.reset()

    def reset(self) -> None:
        self.curr = None
        self.memory.reset_state()

    def reset_turn(self) -> None:
        self.curr = None

    def pick(self) -> int:
        avail = self.game.get_avail()
        if self.curr is None:
            pos = random.choice(avail)
        else:
            x = self.__map(0, self.curr)
            positions = self.memory.infer(x)[: self.dim]
            pos = avail[torch.argmax(positions.flatten()[avail])]
        _, self.curr, cont = self.game.pick(pos)
        h = self.__map(pos, self.curr)
        h = torch.unsqueeze(h, dim=0)
        self.memory.memorize(h)
        if not cont:
            self.reset_turn()
        return pos

    def __map(self, pos: int, label: int, uniform: bool = True) -> torch.Tensor:
        torch.manual_seed(hash((pos, label)))
        return (
            torch.rand(1, self.capacity) if uniform else torch.randn(1, self.capacity)
        )


class BinaryAttractorMemory(BaseStrategy):
    def __init__(
        self,
        game: Game,
        memory: Attractor,
        capacity: int,
        device: str = "cpu",
    ) -> None:
        super().__init__(game)
        self.device = torch.device(device)
        self.memory = memory
        self.memory.to(self.device)

        self.capacity = capacity // 2

        self.reset()

    def reset(self) -> None:
        self.curr = None
        self.memory.reset_state()

    def reset_turn(self) -> None:
        self.curr = None

    def pick(self) -> int:
        avail = self.game.get_avail()
        if self.curr is None:
            pos = random.choice(avail)
        else:
            x = self.__map(0, self.curr).to(self.device)
            # activate all except the ones that indicate position

            positions = self.memory.infer(x)[: self.capacity]
            pos = avail[torch.argmax(positions.flatten()[avail])]
        _, self.curr, cont = self.game.pick(pos)
        h = self.__map(pos, self.curr).to(self.device)
        self.memory.memorize(h)
        if not cont:
            self.reset_turn()
        return pos

    def __map(self, pos: int, label: int) -> torch.Tensor:
        pos_t = self.__binary_tensor(pos + 1)
        label_t = self.__binary_tensor(label)
        return torch.unsqueeze(torch.concat([pos_t, label_t]), dim=0)

    def __binary_tensor(self, n: int) -> torch.Tensor:
        binary_str = format(n % 2**self.capacity, f"0{self.capacity}b")
        return torch.tensor([int(x) for x in binary_str], dtype=torch.float64)


class CNNAttractorMemory(BaseStrategy):
    def __init__(
        self,
        game: Game,
        cvmodel: CVModel,
        memory: Attractor,
        dim: int,
        device: str = "cpu",
    ) -> None:
        super().__init__(game)
        self.device = torch.device(device)
        self.cvmodel = cvmodel
        self.memory = memory
        self.cvmodel.to(self.device)
        self.memory.to(self.device)

        self.dim = dim
        self.cv_out = cvmodel.get_out_features()
        self.block_size = self.cv_out // self.dim

        self.reset()

    def reset(self) -> None:
        self.curr = None
        self.memory.reset_state()

    def reset_turn(self) -> None:
        self.curr = None

    def pick(self) -> int:
        avail = self.game.get_avail()
        if self.curr is None:
            pos = random.choice(avail)
        else:
            unavail = self.game.get_avail(unavailable=True)
            unavail = self.__get_indices(unavail)
            pos_t = torch.zeros(self.cv_out).to(self.device)
            pos_t = torch.unsqueeze(pos_t, dim=0)
            x = torch.concat([pos_t, self.curr], dim=1)
            positions = self.memory.infer(x, unavail).flatten()[: self.cv_out]
            pos_means = positions.unfold(0, self.block_size, self.block_size).mean(
                dim=1
            )
            pos = avail[torch.argmax(pos_means.flatten()[avail])]

        img, _, cont = self.game.pick(pos)
        self.curr = self.cvmodel(img.unsqueeze(0).to(self.device))
        pos_t = torch.zeros(self.cv_out).to(self.device)
        pos_t[self.__get_indices([pos])] = 1
        pos_t = torch.unsqueeze(pos_t, dim=0)
        h = torch.concat([pos_t, self.curr], dim=1)
        self.memory.memorize(h.to(self.device))
        if not cont:
            self.reset_turn()
        return pos

    def __get_indices(self, indices: list[int]) -> list[int]:
        out = np.zeros(self.dim)
        out[indices] = 1
        out = np.repeat(out, self.block_size)
        out.resize(self.cv_out)
        return list(np.where(out)[0])
