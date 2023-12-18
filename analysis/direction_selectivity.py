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
parser.add_argument('--data_dir', default='../data/drifting', help="Directory containing the drifting stimuli")
parser.add_argument('--model_dir', default='../experiments/gamma_forest_ista', help="Directory containing params.json")


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

    X = torch.tensor(np.transpose(np.load(op.join(data_dir, "drifting_10step.npy")), (3, 2, 0, 1)))
    # go through each degree with full batch size
    result_dict = None
    for b in range(X.size(0)):
        data_batch = X[b].repeat(params.batch_size, 1, 1, 1).reshape(params.batch_size, 10, -1).to(device)
        r_d = record(model, data_batch)
        if result_dict is None:
            result_dict = dict()
            for k in r_d.keys():
                result_dict[k] = r_d[k][0:1]
        else:
            for k in result_dict.keys():
                result_dict[k] = torch.cat((result_dict[k], r_d[k][0:1]), axis=0)

    # save second level activations
    save_dir = "results/direction_selectivity/"
    if not op.exists(save_dir):
        os.makedirs(save_dir)
    np.savez(op.join(save_dir, f"result_dict.npz"), **result_dict)
 