from src_old.memory_game.game import Game
from src_old.wm.wm import WorkingMemory
from src_old.wm.agent import Agent

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

n_trials = 1
size_range = [[3,3], [4,4], [5,5], [6,6], [7,7]]

objective = [0.29, 0.38, 0.44, 0.51, 0.58]

game = Game(
            dataset_name='coco-2017', 
            split='validation', 
            field="ground_truth", 
            ds_filter=F("supercategory").is_in(('person', 'animal', 'food', 'vehicle', 'indoor')),
            dataset_dir ="/home/ravi/datasets/coco-2017",
            seed=seed
        )

# FUNCTIONS

def play_strategy(strategy, game, max_steps=200):
    win = False
    steps = 0
    while not win and steps < max_steps:
        strategy.pick()
        win, tot = game.check_win()
        torch.cuda.empty_cache()
        steps+=1
    return tot

def objective_f(x):
    memory = WorkingMemory([50, 50], 4, [x[0], x[1], x[2], x[3], x[4], x[5]], x[6], 0.0, x[7], x[8], seed=seed)
    strategy = Agent(game, memory, seed=seed)
    data = []
    for size in size_range:
        data_size = []
        for n in range(n_trials):
            game.set_size(size)
            tot = play_strategy(strategy, game)
            data_size.append(math.log10(tot/np.prod(size)))
        data.append(data_size)
    results = np.mean(np.array(data), axis=1)
    return ((results - objective)**2).mean()

if __name__ == '__main__':

    with cma.fitness_transformations.EvalParallel2(objective_f, 10) as parallel_obj:
        x, es = cma.fmin2(None, [-0.28, 1.03, 0.05, -0.06, 0.99, 0.05, 0.2, 0.02, 0.02], 0.1, {'bounds': [[-2, 0, 0, -2, 0, 0, 0, 0, 0], [0, 2, 5, 0, 2, 5, 1, 1, 1]], 'verb_disp' : 1}, parallel_objective=parallel_obj)
    print(dict(es.result._asdict()))

    # n_proc = 10
    # es = cma.CMAEvolutionStrategy([-0.28, 1.03, 0.05, -0.06, 0.99, 0.05, 0.2, 0.02, 0.02], 0.01, {'bounds': [[-2, 0, 0, -2, 0, 0, 0, 0, 0], [0, 2, 5, 0, 2, 5, 1, 1, 1]], 'verb_disp' : 1})
    # with cma.fitness_transformations.EvalParallel2(objective_f, n_proc) as eval_para:
    #     while not es.stop():
    #         X = es.ask()
    #         es.tell(X, eval_para(X))

    # print(dict(es.result._asdict()))
