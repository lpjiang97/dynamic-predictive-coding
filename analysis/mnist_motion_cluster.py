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
from evaluation import record_test_set


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='../data/mnist_long', help="Directory containing MNIST dataset")
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

    # test set
    result_dict = record_test_set(params, model, dl, input_dim=324, mixture=True) 
    # save second level activations
    R = result_dict["R_hat"] 
    R2 = result_dict["R2_hat"]
    W = result_dict["W"]
    np.save(op.join(save_dir, "R.npy"), R.numpy())
    np.save(op.join(save_dir, "R2.npy"), R2.numpy())
    np.save(op.join(save_dir, "W.npy"), W.numpy())
    