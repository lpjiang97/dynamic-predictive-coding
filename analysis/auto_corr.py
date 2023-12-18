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
import utils
from evaluation import record


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='../data/forest',
                    help="Directory containing autocorrelation dataset")
parser.add_argument('--model_dir', default='../experiments/gamma_forest_ista',
                    help="Directory containing params.json")
parser.add_argument('--tau', default=10, type=int, help="time lag to compute autocorrelation")


def fit(x, y):
    # init parameter
    p0 = (1., 1., 0.)
    params, cv = curve_fit(exp_decay, x, y, p0, bounds=([-np.inf, -np.inf, 0], np.inf))
    return params


def exp_decay(x, m, t, b):
    return m * np.exp(-t * x) + b


def autocorr(response, tau):
    batch_size = response.shape[0]
    dim = response.shape[2]
    r = np.zeros((batch_size, dim, tau))
    for b in range(batch_size):
        for n in range(dim):
            r[b, n] = acf(response[b, 1:, n], nlags=tau-1)
    # take mean and std
    r = r.reshape(-1, tau)
    r_mean = np.nanmean(r, 0)
    r_std = np.nanstd(r, 0)
    # fit exp decay
    params = fit(np.arange(10), r_mean)
    # simulate using params
    x = np.linspace(0, 10, 100)
    rhat = exp_decay(x, *params)
    # dict
    keys = ["mean", "std", "params", "rhat", "raw"]
    results = [r_mean, r_std, params, rhat, r]
    d = {keys[i]:results[i] for i in range(len(results))}
    return d


def response_autocorr(model, input_batch, tau, save_dir, mode):
    # I_hat, I, I_star, R_hat, #R, R2, S1, S2, W, ALPHA, BETA, r2_losses = inf_and_record(model, input_batch, N=10)
    result_dict = record(model, input_batch, input_dim=256)
    R = result_dict["R_hat"] 
    R2 = result_dict["R2_hat"]
    W = result_dict["W"]
    r_dict  = autocorr(R, tau)
    rh_dict = autocorr(R2, tau)
    w_dict = autocorr(W, tau)
    # save
    np.savez(op.join(save_dir, f"r_dict_{mode}.npz"), **r_dict)
    np.savez(op.join(save_dir, f"rh_dict_{mode}.npz"), **rh_dict)
    np.savez(op.join(save_dir, f"w_dict_{mode}.npz"), **w_dict)
    np.save(op.join(save_dir, f"raw_r_{mode}"), r_dict["raw"])
    np.save(op.join(save_dir, f"raw_rh_{mode}"), rh_dict["raw"])


if __name__ == '__main__':

    args = parser.parse_args()
    # load data and model
    fpath = args.model_dir
    data_dir = args.data_dir
    params = utils.Params(op.join(fpath, 'params.json'))
    params.cuda = torch.cuda.is_available()
    params.shuffle = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = DynPredNet(params, device).to(device)
    ckpt = utils.load_checkpoint(op.join(fpath, 'last.pth.tar'), model)
    model.eval()
    save_dir = op.join("results", "autocorr")
    if not op.exists(save_dir):
        os.makedirs(save_dir)
 
    batch_size = params.batch_size
    time_length = 100
    tau = args.tau
    # white noise stimui
    input_batch = torch.randn(batch_size, time_length, 256) * 0.05
    input_batch = input_batch.to(device)
    response_autocorr(model, input_batch, tau, save_dir, "white")
    print("White noise done")
    # natural stimuli
    input_batch = torch.tensor(np.load(op.join(data_dir, "autocorr.npy"))).to(device)[:,:time_length]
    response_autocorr(model, input_batch, tau, save_dir, "nat")
    print("Natural done")
