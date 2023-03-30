import random

import math

import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.core.dataset as ds

from PIL import Image

from torchvision import transforms

import torch

import numpy as np


class Game:
    def __init__(
        self,
        dataset_name: str,
        split: list[str] = None,
        dataset_dir: str = None,
        field: str = None,
        ds_filter: fo.core.expressions.ViewExpression = None,
        image_size: tuple[int] = (480, 480),
        grayscale: bool = False,
        card_back_color=128,
        grid_size: list[int] = [3, 3],
        n_matching: int = 2,
    ) -> None:
        self.grid_size = grid_size
        self.n_elements = np.prod(grid_size)
        self.n_matching = n_matching

        self.__load_dataset(dataset_name, split, dataset_dir, field, ds_filter)
        self.image_size = image_size
        self.grayscale = grayscale
        self.card_back_color = card_back_color

        self.reset()

    def pick(self, pos: int) -> tuple[torch.Tensor, int, bool]:
        if pos in self.revealed or self.flipped[pos]:
            raise Exception("Invalid index")
        self.counter += 1
        self.__reveal_card(pos)
        if len(self.revealed) < self.n_matching and len(set(self.revealed_lab)) <= 1:
            return self.grid[pos], self.grid_labels[pos], True
        elif len(self.revealed) == self.n_matching and len(set(self.revealed_lab)) == 1:
            self.flipped[self.revealed] = True
        self.__reset_turn()
        return self.grid[pos], self.grid_labels[pos], False

    def check_win(self) -> tuple[bool, int]:
        return self.n_elements - sum(self.flipped) < self.n_matching, self.counter

    def reset(self) -> None:
        self.flipped = np.full(self.n_elements, False)
        self.revealed = []
        self.revealed_lab = []
        self.counter = 0
        self.__build_grid()

    def __reset_turn(self) -> None:
        self.revealed = []
        self.revealed_lab = []

    def __reveal_card(self, pos: int) -> None:
        self.revealed.append(pos)
        self.revealed_lab.append(self.grid_labels[pos])

    def __build_grid(self) -> None:
        n_elements = self.n_elements
        elements = np.random.choice(
            len(self.dataset), math.ceil(n_elements / self.n_matching), replace=False
        )
        elements = np.repeat(elements, self.n_matching)[:n_elements]
        np.random.shuffle(elements)
        self.grid = self.__get_images(elements)
        self.grid_labels = elements

    def __load_dataset(
        self,
        dataset_name: str,
        split: list[str] = None,
        dataset_dir: str = None,
        field: str = None,
        ds_filter: fo.core.expressions.ViewExpression = None,
    ) -> None:
        ds = foz.load_zoo_dataset(dataset_name, split=split, dataset_dir=dataset_dir)
        self.dataset = self.__filter_ds(ds, field, ds_filter) if field else ds
        self.img_paths = self.dataset.values("filepath")

    def __filter_ds(self, dataset, field, filter):
        return dataset.filter_labels(field, filter)

    def __get_images(self, slice: np.ndarray) -> torch.Tensor:
        return torch.stack([self.__get_image(idx) for idx in slice])

    def __get_image(self, idx: int) -> torch.Tensor:
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("RGB")
        t = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize(self.image_size, antialias=True)]
        )
        gs = transforms.Grayscale()
        img = t(img)
        return gs(img) if self.grayscale else img

    def get_grid(self, flat=True) -> np.ndarray:
        grey = torch.full_like(self.grid, 128)
        grey[self.revealed] = self.grid[self.revealed]
        return grey if flat else grey.reshape(self.grid_size)

    def get_grid_labels(self, flat=True) -> np.ndarray:
        return self.grid_labels if flat else self.grid_labels(self.grid_size)

    def get_avail(self) -> list[int]:
        return list(set(np.where(self.flipped == False)[0]) - set(self.revealed))

    def get_revealed(self) -> list[int]:
        return self.revealed.copy()

    def set_size(self, grid_size: int) -> None:
        self.grid_size = grid_size
        self.n_elements = np.prod(grid_size)
        self.reset()
