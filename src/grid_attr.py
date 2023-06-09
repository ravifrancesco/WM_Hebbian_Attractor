from game import Game
from strategies import FastAttractorMemory
from memory import FastAttractor

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

n_trials = 20
size_range = [[3,3], [4,4], [5,5], [6,6], [7,7]]
n_matching = 2

steps = 10

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
            dataset_dir ="/home/ravi/datasets/coco-2017"
        )

# FUNCTIONS

def nmse(x, y):
    # mms = MinMaxScaler()
    # x_n = mms.fit_transform(np.array(x).reshape(-1,1))
    # y_n = mms.fit_transform(np.array(y).reshape(-1,1))
    return mean_squared_error(x, y)
    

def play_strategy(strategy, game, max_steps=500):
    win = False
    steps = 0
    while not win and steps < max_steps:
        strategy.pick()
        win = game.check_win()
        torch.cuda.empty_cache()
        steps+=1

def objective_f(x):
    dfs = []
    tot_clicks = []
    for size in size_range:
        size_clicks = []
        for n in range(n_trials):
            game.set_size(size, n_matching)
            dim = np.prod(size) + len(np.unique(game.get_grid_labels()))
            # dim = 74
            memory = FastAttractor(dim, x[0], x[1])
            strategy = FastAttractorMemory(game, memory, np.prod(size), steps=steps)
            play_strategy(strategy, game)
            df, nc = game.get_metrics()
            dfs.append(df)
            size_clicks.append(nc)
        tot_clicks.append(size_clicks)
    
    df = pd.concat(dfs).reset_index(drop=True)
    m_df = pd.DataFrame(df[df['match']==1]).groupby('board_size')
    mm_df = pd.DataFrame(df[df['match']==0]).groupby('board_size')
    
    nc_means = np.mean(np.array(tot_clicks), axis=1)
    mse_nc = nmse(nc_means, nc_obj)

    mse_nslc_m = nmse(m_df.mean()['nslc'].to_numpy(), nslc_m_obj)
    mse_nslc_mm = nmse(mm_df.mean()['nslc'].to_numpy(), nslc_mm_obj)
    
    mse_nslp_m = nmse(m_df.mean()['nslp'].to_numpy(), nslp_m_obj)
    mse_nslp_mm = nmse(mm_df.mean()['nslp'].to_numpy(), nslp_mm_obj)
    
    return mse_nslc_m + mse_nslc_mm

if __name__ == '__main__':

    # # with cma.fitness_transformations.EvalParallel2(objective_f, 5) as parallel_obj:
    # x, es = cma.fmin2(objective_f, [0.5, 0.5], 0.05, {'bounds': [[0,0], [1,1]], 'verb_disp' : 1})#, parallel_objective=parallel_obj)
    # print(dict(es.result._asdict()))
    # print('This was done with rand init, learning in loop, fixed steps at 100, -1 on unavailable, fit on second graph')

    grid_lr = np.linspace(0.05, 1, 20)
    grid_rr = np.linspace(0.05, 1, 20)

    grid_res = []
    for lr in tqdm(grid_lr, desc='LR'):
        grid_lr = []
        for rr in tqdm(grid_rr, desc='RR', leave=False):
            grid_lr.append(objective_f([lr, rr]))
        grid_res.append(grid_lr)
    
    grid_arr = np.array(grid_res)
    plt.figure() 
    ax = sns.heatmap(grid_arr, annot=False, cmap='coolwarm', square=True, cbar=True, cbar_kws={'label': 'MSE'}, xticklabels=np.linspace(0.05, 1, 20), yticklabels=np.linspace(0.05, 1, 20))
    ax.set(xlabel="RR", ylabel="LR")
    plt.show()
    plt.savefig("/home/ravi/Models-of-Working-Memory/src/figures/grid_nslc.svg")
    

    

