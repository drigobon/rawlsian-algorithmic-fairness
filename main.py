import sys
import numpy as np
import pandas as pd

import json

from utils import *

'''
# Args
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-l", "--layers", type = int, help="number of hidden layers")
parser.add_argument("-m", "--models", type = int, help="number of randomized models")
args = parser.parse_args()


layers = args.layers
n_models = args.models
'''


# Parameters

l_min = -1   # in powers of 10
l_max = 1.3    # in powers of 10
n_l = 100
n_models = 20
n_layers = 0


# Whether or not we use the whole set of samples as a single batch

full_batch = True


datasets = ['COMPAS', 'BANK', 'ADULT', 'DEFAULT', 'COMMUNITIES']


for dataset in datasets:

    if dataset == 'COMPAS':
        X,y = load_compas()

    elif dataset == 'BANK':
        X,y = load_bank()

    elif dataset == 'ADULT':
        X,y = load_adult()

    elif dataset == 'DEFAULT':
        X,y = load_default()

    elif dataset == 'COMMUNITIES':
        X,y = load_communities()


    sys.stdout.write('\n-------------------------------------- \n '+str(dataset)+'\n')
    sys.stdout.flush()


    df_out = get_frontier(X, y, l_min, l_max, n_l, 
                         n_models = n_models, 
                         n_layers = layers,
                         dataset = dataset,
                         include_min_mse = True,
                         full_batch = full_batch,
                         )


    filename = 'Output/'+dataset+'-'+str(layers)+'-Layer-NN'

    df_out.to_csv(filename+'.csv')


    params = {
              'l_min': l_min, 'l_max': l_max, 'n_l': n_l, 
              'n_models': n_models,
              }

    with open(filename+'.json', 'w') as fp:
        json.dump(params, fp)
