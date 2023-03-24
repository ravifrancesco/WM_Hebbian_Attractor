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
        size: int,
        n_matching: int,
        dataset_name: str,
        splits: list[str] = None,
        dataset_dir: str = None,
        field: str = None,
        ds_filter: fo.core.expressions.ViewExpression = None,
        image_size: tuple[int] = (480, 480),
    ) -> None:
        self.size = size
        self.n_elements = size**2
        self.n_matching = n_matching

        self.__load_dataset(dataset_name, splits, dataset_dir, field, ds_filter)
        self.image_size = image_size

        self.reset_game()

    def reset_game(self) -> None:
        self.flipped = np.full(self.n_elements, False)
        self.available = np.arange(self.n_elements)
        self.revealed = []
        self.revealed_lab = []
        self.__build_grid()

    def pick(self, idx: int = None) -> tuple[torch.Tensor, int, bool]:
        if idx in self.revealed or self.flipped[idx]:
            print("Invalid index")
            return None
        avail = np.array(set(self.available) - set(self.revealed))
        pos = idx if idx is not None else np.random.choice(avail)
        self.__reveal_card(pos)
        if len(self.revealed) < self.n_matching and len(set(self.revealed_lab)) <= 1:
            return self.grid[pos], self.grid_labels[pos], True
        elif len(self.revealed) == self.n_matching and len(set(self.revealed_lab)) == 1:
            self.flipped[self.revealed] = True
        self.__reset_turn()
        return self.grid[pos], self.grid_labels[pos], False

    def check_win(self) -> bool:
        return self.n_elements - sum(self.flipped) < self.n_matching

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

    def __get_images(self, slice: np.ndarray) -> np.ndarray:
        return np.array([self.__get_image(idx) for idx in slice])

    def __get_image(self, idx: int, grayscale=False) -> torch.Tensor:
        img_path = self.img_paths[idx]
        img = Image.open(img_path)
        t = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize(self.image_size)]
        )
        img = t(img)
        return transforms.Grayscale(img) if grayscale else img

    def __load_dataset(
        self,
        dataset_name: str,
        splits: list[str] = None,
        dataset_dir: str = None,
        field: str = None,
        ds_filter: fo.core.expressions.ViewExpression = None,
    ) -> None:
        ds = foz.load_zoo_dataset(dataset_name, splits, dataset_dir)
        self.dataset = self.__filter_ds(ds, field, ds_filter) if field else ds
        self.img_paths = self.dataset.values("filepath")

    def __filter_ds(self, dataset, field, filter):
        return dataset.filter_labels(field, filter)

    def set_seed(self, seed: int = 42) -> None:
        np.random.seed(seed)
