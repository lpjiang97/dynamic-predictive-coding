import argparse
import os
import os.path as op
import sys
sys.path.insert(0, op.abspath("../."))

import numpy as np
import torch
from tqdm import tqdm

import utils
from models.predictive_coding_ista import DynPredNet


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='../experiments/gamma_forest_ista')


def reverse_corr(model, stimuli, save_dir, T=10):
    # stimuli: B x T x S
    batch_size, length, N = stimuli.shape
    K = model.spatial_decoder.weight.shape[1]
    dev = model.device
    stim = torch.zeros((T, N, batch_size, K), device=dev)
    # fit fi
    r, r2 = model.init_code_(batch_size)
    # first step p(r_0, s_0 | I_0)
    x = stimuli[:, 0].to(dev)
    r  = model.inf_first_step(x)
    # reverse corr stimuli
    for t in tqdm(range(1, length), total=length, desc="Reverse corr"):
        # inference
        r_p = r.clone().detach()
        # prediction
        r_hat = model.temporal_prediction_(r_p, r2).clone().detach()
        x = stimuli[:, t].to(dev)
        r, r2, _ = model.inf(x, r_p, r2.clone().detach())
        # any instability, reinit
        if torch.any(torch.isnan(r)):
            print("nan detected")
            r, r2 = model.init_code_(batch_size)
        # record weighted average
        if t > T+1:
            # past stim: Time x RF size x Batch x Num Neuron
            past_stim = stimuli[:, t-T-1:t-1].permute(1, 2, 0).reshape(T, N, batch_size, 1).repeat(1, 1, 1, K)
            past_stim = past_stim.to(dev)
            # save
            stim += past_stim * r_hat
        # save checkpoints
        if t % 10000 == 9999:
            strf = stim.cpu().numpy()
            np.save(op.join(save_dir, f"STRF_{t+1}"), strf)

    return stim


if __name__ == '__main__':

    args = parser.parse_args()
    # load data and model
    fpath = args.model_dir
    params = utils.Params(op.join(fpath, 'params.json'))
    params.cuda = torch.cuda.is_available()
    params.shuffle = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # predictive coding network
    prednet = DynPredNet(params, device)
    ckpt = utils.load_checkpoint(op.join(fpath, 'last.pth.tar'), prednet)
    prednet = prednet.to(device)
    prednet.eval()

    # load reverse correlation stimuli
    stimuli = torch.tensor(np.load("../data/forest/reverse_corr.npy"))

    save_dir = "results/strf"
    if not op.exists(save_dir):
        os.makedirs(save_dir)

    STRF = reverse_corr(prednet, stimuli, save_dir)
    STRF = STRF.cpu().numpy()
    np.save(op.join(save_dir, "STRF_final"), STRF)
    # save just spatial RF
    np.save(op.join(save_dir, "RF"), prednet.spatial_decoder.weight.cpu().detach().numpy())
