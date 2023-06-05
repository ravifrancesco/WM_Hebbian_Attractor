from src.memory_game.game import Game
from src.strategy.strategies import BaseStrategy, PerfectMemory, TileMemory, RandomHashMemory, BaseAttractorMemory, CNNAttractorMemory, RandomAttractorMemory, BernoulliMemory, BinaryAttractorMemory, FastAttractorMemory
from src.models.cvmodel import CVModel
from src.models.memory import TileRNN, HashRNN, Attractor, FastAttractor

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from fiftyone import ViewField as F

import math

import torch

import pandas as pd

import numpy as np

import collections, gc, resource

from tqdm import tqdm

import seaborn as sns

import cma

import matplotlib.pyplot as plt
import matplotlib.gridspec as grd
sns.set_style(style='white')

torch.no_grad()

# GLOBAL VARIABLES


seed = 42

n_trials = 20
size_range = [[3,3], [4,4], [5,5], [6,6], [7,7]]

nc_obj = [0.28, 0.38, 0.44, 0.5, 0.58]
nslc_m_obj = [5.48, 6.66, 7.16, 9.09, 12.15]
nslc_mm_obj = [7.41, 9.55, 14.95, 19.27, 29.53]
nslp_m_obj = [1., 1., 1., 1., 1.]
nslp_mm_obj = [6.45, 9.14, 13.15, 22.02, 29.49]

game = Game(
            dataset_name='coco-2017', 
            split='validation', 
            field="ground_truth", 
            ds_filter=F("supercategory").is_in(('person', 'animal', 'food', 'vehicle', 'indoor')),
            dataset_dir ="/home/ravi/datasets/coco-2017",
            seed=seed
        )

# FUNCTIONS

def nmse(x, y):
    mms = MinMaxScaler()
    x_n = mms.fit_transform(np.array(x).reshape(-1,1))
    y_n = mms.fit_transform(np.array(y).reshape(-1,1))
    return mean_squared_error(x_n, y_n)
    

def play_strategy(strategy, game, max_steps=500):
    win = False
    steps = 0
    while not win and steps < max_steps:
        strategy.pick()
        win, tot = game.check_win()
        torch.cuda.empty_cache()
        steps+=1
    return tot

def objective_f(x):
    data = []
    for size in size_range:
        for n in range(n_trials):
            game.set_size(size)
            dim = np.prod(size) + len(np.unique(game.get_grid_labels()))
            # dim = 74
            memory = FastAttractor(dim, x[0], x[1])
            strategy = FastAttractorMemory(game, memory, np.prod(size), steps=int(x[2]))
            tot = play_strategy(strategy, game)
            tot = math.log10(tot/np.prod(size))
            nlsc_m, nlsc_mm = game.get_number_since_last_click()
            nlsp_m, nlsp_mm = game.get_number_since_last_pair()
            data.append({'run': n, 'size': str(size[0]), 'nc': tot, 'nlsc_m': nlsc_m, 'nlsc_mm': nlsc_mm, 'nlsp_m': nlsp_m, 'nlsp_mm': nlsp_mm})
    means = pd.DataFrame(data).groupby('size').mean()
    mse_nc = nmse(means['nc'].to_numpy(), nc_obj)
    mse_nslc_m = nmse(means['nlsc_m'].to_numpy(), nslc_m_obj)
    mse_nslc_mm = nmse(means['nlsc_mm'].to_numpy(), nslc_mm_obj)
    mse_nslp_m = nmse(means['nlsp_m'].to_numpy(), nslp_m_obj)
    mse_nslp_mm = nmse(means['nlsp_mm'].to_numpy(), nslp_mm_obj)
    return mse_nslp_m + mse_nslp_mm

if __name__ == '__main__':

    # with cma.fitness_transformations.EvalParallel2(objective_f, 5) as parallel_obj:
    x, es = cma.fmin2(objective_f, [0.5, 0.5, 20], 0.1, {'bounds': [[0,0,0], [1,1,100]], 'verb_disp' : 1, 'integer_variables': [2]})#, parallel_objective=parallel_obj)
    print(dict(es.result._asdict()))
