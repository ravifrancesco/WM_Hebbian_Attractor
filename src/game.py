import math

import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.core.dataset as ds

from sklearn import preprocessing

from PIL import Image

from torchvision import transforms

import torch

import numpy as np

import pandas as pd


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
        board_size: list[int] = [3, 3],
        n_matching: int = 2,
        seed: int = None,
    ) -> None:
        if seed is not None:
            np.random.seed(seed)

        self.__load_dataset(dataset_name, split, dataset_dir, field, ds_filter)
        self.image_size = image_size
        self.grayscale = grayscale
        self.card_back_color = card_back_color

        self.set_size(board_size, n_matching)

    def pick(self, pos: int) -> tuple[torch.Tensor, int, bool]:
        if not self.get_avail()[pos]:
            raise Exception("Invalid index")
        self.__reveal_card(pos)
        n_revealed = np.count_nonzero(self.revealed)
        n_revealed_labels = np.count_nonzero(self.revealed_lab)
        if n_revealed < self.n_matching and n_revealed_labels <= 1:
            self.gameplay.update(pos)
            return self.grid[pos], self.grid_labels[pos], True
        elif n_revealed == self.n_matching and n_revealed_labels == 1:
            self.flipped[self.revealed] = True
            self.score += 1
        self.gameplay.update(pos)
        self.__reset_turn()
        return self.grid[pos], self.grid_labels[pos], False

    def check_win(self) -> bool:
        return self.score == self.max_score

    def reset(self) -> None:
        self.__build_grid()
        self.flipped = np.full(self.n_tiles, False)
        self.revealed = np.full(self.n_tiles, False)
        self.revealed_lab = np.full(self.n_labels, False)
        self.score = 0
        self.gameplay = Gameplay(self)

    def __reset_turn(self) -> None:
        self.revealed[:] = False
        self.revealed_lab[:] = False

    def __reveal_card(self, pos: int) -> None:
        self.revealed[pos] = True
        self.revealed_lab[self.grid_labels[pos]] = True

    def __build_grid(self) -> None:
        n_elements = self.n_tiles
        elements = np.random.choice(len(self.dataset), self.n_labels, replace=False)
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

    def get_metrics(self) -> tuple[pd.DataFrame, int]:
        return self.gameplay.get_metrics()

    def get_board_size(self) -> list[int]:
        return self.board_size

    def get_grid(self, flat=True) -> np.ndarray:
        grey = torch.full_like(self.grid, 128)
        grey[self.revealed] = self.grid[self.revealed]
        return grey if flat else grey.reshape(self.board_size)

    def get_grid_labels(self, flat=True) -> np.ndarray:
        return self.grid_labels if flat else self.grid_labels(self.board_size)

    def get_avail(self) -> np.ndarray:
        return ~(self.flipped | self.revealed)

    def get_revealed(self) -> np.ndarray:
        return self.revealed

    def get_flipped(self) -> np.ndarray:
        return self.flipped

    def get_score(self) -> int:
        return self.score

    def get_n_labels(self) -> int:
        return self.n_labels

    def set_size(self, board_size: int, n_matching: int) -> None:
        self.board_size = board_size
        self.n_tiles = np.prod(board_size)
        self.n_matching = n_matching
        self.n_labels = math.ceil(self.n_tiles / self.n_matching)
        self.max_score = self.n_tiles // self.n_matching
        self.reset()


class Gameplay:
    def __init__(self, game: Game) -> None:
        self.game = game
        self.board_size = game.get_board_size()
        self.grid = game.get_grid_labels()

        self.tot_clicks = 0
        self.score = [0]
        self.tile_clicked = []
        self.seen_label = []

    def update(self, pos: int):
        self.tot_clicks += 1
        self.score.append(self.game.get_score())
        self.tile_clicked.append(pos)
        self.seen_label.append(self.grid[pos])

    def get_metrics(self) -> tuple[pd.DataFrame, int]:
        s = np.array(self.score)
        tc = np.array(self.tile_clicked)
        sl = np.array(self.seen_label)
        df = pd.DataFrame(
            {
                "match": self.__build_match(s),
                "tile_clicked": tc,
                "nslc": self.__build_nslc(tc),
                "nslp": self.__build_nslp(tc, sl),
                "correct_tile": self.__build_correct_tile(tc, sl),
            }
        )
        # TODO change later, include >2 different dimensions
        df["board_size"] = np.prod(self.board_size)
        return df, self.tot_clicks

    def __build_match(self, s: np.ndarray) -> np.ndarray:
        diff_array = np.diff(s[0::2])
        match_array = (diff_array > 0).astype(int)
        return np.repeat(match_array, 2)

    def __build_nslc(self, tc: np.ndarray) -> np.ndarray:
        nlsc = []
        for i, v in enumerate(tc):
            idxs = np.where(tc[:i] == v)[0]
            nlsc.append(i - idxs[-1] if idxs.size else -1)
        return np.array(nlsc)

    def __build_nslp(self, tc: np.ndarray, sl: np.ndarray) -> np.ndarray:
        nlsp = []
        for i, v in enumerate(sl):
            idxs = np.where(sl[:i] == v)[0]
            idxs = idxs[tc[idxs] != tc[i]]
            nlsp.append(i - idxs[-1] if idxs.size else -1)
        return np.array(nlsp)

    def __build_correct_tile(self, tc: np.ndarray, sl: np.ndarray) -> np.ndarray:
        ct = []
        for i, v in enumerate(sl):
            idxs = np.where(sl[:i] == v)[0]
            idxs = idxs[tc[idxs] != tc[i]]
            ct.append(tc[idxs[0]] if idxs.size else -1)
        return np.array(ct)
