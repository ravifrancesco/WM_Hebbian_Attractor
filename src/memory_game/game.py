import math

import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.core.dataset as ds

from sklearn import preprocessing

from PIL import Image

from torchvision import transforms

import torch

import numpy as np

# TODO fix metrics, maybe use observer classs or object

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
        seed: int = None,
    ) -> None:
        if seed is not None:
            np.random.seed(seed)

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
        self.nslc = np.where(self.nslc >= 0, self.nslc + 1, self.nslc)
        self.__reveal_card(pos)
        match_pos = self.__get_pair_index(self.grid_labels[pos], pos)
        if len(self.revealed) < self.n_matching and len(set(self.revealed_lab)) <= 1:
            return self.grid[pos], self.grid_labels[pos], True
        elif len(self.revealed) == self.n_matching and len(set(self.revealed_lab)) == 1:
            if self.nslc[pos] > 0:
                self.nslc_match.append(self.nslc[pos])
            if match_pos != -1 and self.nslc[match_pos] > 0:
                self.nslp_match.append(self.nslc[match_pos])
            self.flipped[self.revealed] = True
        else:
            if self.nslc[pos] > 0:
                self.nslc_mismatch.append(self.nslc[pos])
            if match_pos != -1 and self.nslc[match_pos] > 0:
                self.nslp_mismatch.append(self.nslc[match_pos])
        self.nslc[pos] = 0
        self.__reset_turn()
        return self.grid[pos], self.grid_labels[pos], False

    def check_win(self) -> tuple[bool, int]:
        return self.n_elements - sum(self.flipped) < self.n_matching, self.counter

    def reset(self) -> None:
        self.flipped = np.full(self.n_elements, False)
        self.revealed = []
        self.revealed_lab = []
        self.counter = 0
        # Number since last click
        self.nslc = np.full(self.n_elements, -1)
        self.nslc_match = []
        self.nslc_mismatch = []
        # Number since last pair
        self.nslp_match = []
        self.nslp_mismatch = []
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
        self.grid_labels = preprocessing.LabelEncoder().fit_transform(elements)

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
    
    def __get_pair_index(self, label, index):
        matching_indices = np.where(self.grid_labels == label)[0]
        matching_indices = matching_indices[matching_indices != index]
        return matching_indices[0] if matching_indices else -1

    def get_grid(self, flat=True) -> np.ndarray:
        grey = torch.full_like(self.grid, 128)
        grey[self.revealed] = self.grid[self.revealed]
        return grey if flat else grey.reshape(self.grid_size)

    def get_grid_labels(self, flat=True) -> np.ndarray:
        return self.grid_labels if flat else self.grid_labels(self.grid_size)

    def get_avail(self, unavailable=False) -> list[int]:
        if unavailable:
            return list(set(np.where(self.flipped == True)[0]) | set(self.revealed))
        else:
            return list(set(np.where(self.flipped == False)[0]) - set(self.revealed))

    def get_revealed(self) -> list[int]:
        return self.revealed

    def get_flipped(self) -> list[int]:
        return np.where(self.flipped)[0]
    
    def get_number_since_last_click(self) -> tuple[float, float]: # TODO clean
        return 0.0 if not self.nslc_match else np.mean(self.nslc_match), 0.0 if not self.nslc_mismatch else np.mean(self.nslc_mismatch)
    
    def get_number_since_last_pair(self) -> tuple[float, float]: # TODO clean
        return 0.0 if not self.nslp_match else np.mean(self.nslp_match), 0.0 if not self.nslp_mismatch else np.mean(self.nslp_mismatch)

    def set_size(self, grid_size: int) -> None:
        self.grid_size = grid_size
        self.n_elements = np.prod(grid_size)
        self.reset()
