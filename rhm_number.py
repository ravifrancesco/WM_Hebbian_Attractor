from src_old.memory_game.game import Game
from src_old.strategy.strategies import BaseStrategy, PerfectMemory, TileMemory, RandomHashMemory
from src_old.models.cvmodel import CVModel
from src_old.models.memory import TileRNN, HashRNN

from fiftyone import ViewField as F

import math

import torch

import pandas as pd

import numpy as np

from tqdm import tqdm

import seaborn as sns
sns.set_style(style='white')

import matplotlib.pyplot as plt
import matplotlib as mpl

torch.no_grad()

PATH = "/home/ravi/figures/rhm_lstm/"

def play_strategy(strategy, game):
    win = False
    while not win:
        strategy.pick()
        win, tot = game.check_win()
    return tot

def collect_data_limited_cap(n_trials, size_range, cvmodel, game, cap, input_size=1000, distance_metric='manhattan'):
    data = []
    for size in tqdm(size_range, desc=f'Size: {cap}'):
        data_size = []
        for n in range(n_trials):
            game.set_size(size)
            memory = HashRNN('lstm', 10, 1000, 1000)
            strategy = RandomHashMemory(game, cvmodel, memory, distance_metric='manhattan', device=device)
            tot = play_strategy(strategy, game)
            data_size.append(math.log10(tot/np.prod(size)))
        data.append(data_size)
    return data

def plot_data(data):
    plt.figure(figsize=(6, 6))
    cmap = plt.get_cmap('mako')
    ax = sns.lineplot(x="size", y='nclicks', hue="cap", data=data, errorbar=None, palette=cmap, legend=False)

    cbar = ax.figure.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=data['cap'].min(),
                                                                            vmax=data['cap'].max(),
                                                                            clip=False),
                                                    cmap=cmap),
                            ticks=np.arange(data['cap'].min(), data['cap'].max() + 1, 5),
                            label=r'Number of cells (LSTM - 1000 hidden units)')
    cbar.ax.tick_params(size=0)
    # # cbar.ax.invert_yaxis()  # optionally invert the yaxis of the colorbar
    # # ax.legend_.remove()  # for testing purposes don't yet remove the legend
    ax.set_xlabel("Board size")
    ax.set_ylabel("N clicks per tile (log10)")
    plt.tight_layout()
    plt.show()
    plt.savefig(f'{PATH}fig.svg')

if __name__ == '__main__':

    print(f'Cuda available: {torch.cuda.is_available()}')
    print(f'Cuda device count: {torch.cuda.device_count()}')

    device = 1

    device = torch.device(f"cuda:{device}")
    print(torch.cuda.get_device_name(device))

    game = Game(
            dataset_name='coco-2017', 
            split='validation', 
            field="ground_truth", 
            ds_filter=F("supercategory").is_in(('person', 'animal', 'food', 'vehicle', 'indoor')),
            dataset_dir ="/home/ravi/datasets/coco-2017",
        )

    n_trials = 20
    size_range = [[3,3], [4,4], [5,5], [6,6], [7,7]]

    cvmodel = CVModel('pytorch/vision:v0.10.0', 'resnet18', model_dir='/home/ravi/models')

    data = []
    for cap in range(5, 51, 1):
        cap_data = collect_data_limited_cap(n_trials, size_range, cvmodel, game, cap=cap)
        cap_data = pd.melt(pd.DataFrame(cap_data).T, var_name='size', value_name='nclicks')
        cap_data['cap'] = cap
        data.append(cap_data)

    df = pd.concat(data)
    df['size']+=3
    df['size'] = df['size'].astype('str')
    df.to_csv(f'{PATH}data.csv')
    plot_data(df)
