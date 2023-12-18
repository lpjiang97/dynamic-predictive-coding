import argparse
import os
import os.path as op
import sys
sys.path.insert(0, os.path.abspath('../.'))

import numpy as np
from scipy.optimize import curve_fit
import torch
from statsmodels.tsa.stattools import acf

from models.predictive_coding_ista import DynPredNet
import models.data_loader as data_loader
import utils
from evaluation import record


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='../data/mnist_long', help="Directory containing the MNIST dataset")
parser.add_argument('--model_dir', default='../experiments/mnist_ista', help="Directory containing params.json")


if __name__ == '__main__':

    args = parser.parse_args()
    # load data and model
    fpath = args.model_dir
    data_dir = args.data_dir
    params = utils.Params(op.join(fpath, 'params.json'))
    params.cuda = torch.cuda.is_available()
    # keep order the same
    params.shuffle = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = DynPredNet(params, device).to(device)
    ckpt = utils.load_checkpoint(op.join(fpath, 'last.pth.tar'), model)
    model.eval()
    save_dir = op.join("results", "mnist")
    if not op.exists(save_dir):
        os.makedirs(save_dir)
 
    batch_size = params.batch_size
    dataloaders = data_loader.fetch_dataloader(['test'], args.data_dir, params)
    dl = dataloaders['test']

    X = dl.dataset.data[:params.batch_size].to(device)
    
    save_dir = "results/inf_example/"
    if not op.exists(save_dir):
        os.makedirs(save_dir)

    # inference example
    result_dict = record(model, X, input_dim=params.input_dim, mixture=True, turnoff=10)
    np.savez(op.join(save_dir, f"result_dict_mnist.npz"), **result_dict)
    
    # long term prediction 
    result_dict = record(model, X, input_dim=params.input_dim, mixture=True, turnoff=3)
    np.savez(op.join(save_dir, f"result_dict_long_pred_mnist.npz"), **result_dict)
 