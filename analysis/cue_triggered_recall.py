import argparse
import os
import os.path as op
import sys
sys.path.insert(0, op.abspath("../."))
import copy

import numpy as np
from scipy import special
from scipy.signal import correlate, correlation_lags
import torch

import utils
import models.data_loader as data_loader
from models.memory import Memory, corrupt_input
from models.predictive_coding_ista import DynPredNet


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='../data/memory',
                    help="Directory containing the dataset")
parser.add_argument('--model_dir', default='../experiments/memory',
                    help="Directory containing params.json")


def xcorr(U, R, size=18, avg=False):
    U_loc = np.zeros((U.shape[0], 2))
    for i in range(U.shape[0]):
        U_loc[i] = U[i].argmax() % size, U[i].argmax() // size
    # pair-wise distance
    pairwise_dist = np.zeros((int(special.comb(U.shape[0], 2))))
    c = 0
    for i in range(U.shape[0]):
        for j in range(i):
            #pairwise_dist[c] = np.abs(RF_dist[i] - RF_dist[j])
            #pairwise_dist[c] = np.abs(U_loc[i, 0] - U_loc[j, 0])
            pairwise_dist[c] = np.sqrt(np.sum(((U_loc[i] - U_loc[j]) ** 2)))
            c += 1
    # pair-wise corr
    lags = correlation_lags(len(R[0]), len(R[1]))
    pairwise_xcorr = np.zeros((pairwise_dist.shape[0], len(lags)))
    c = 0
    for i in range(U.shape[0]):
        for j in range(i):
            pairwise_xcorr[c] = correlate(R[i], R[j])
            c += 1
    # sort
    idx = np.argsort(pairwise_dist)
    pairwise_dist_sorted = pairwise_dist[idx]
    pairwise_xcorr_sorted = pairwise_xcorr[idx]
    if avg:
        # split to blocks
        blocks = []
        d = 0
        prev = 0
        for i in range(pairwise_dist_sorted.shape[0]):
            if pairwise_dist_sorted[i] - d > 0.5:
                blocks.append(pairwise_xcorr_sorted[prev:i])
                prev = i
                d += 1
        # average in blocks
        for i in range(len(blocks)):
            blocks[i] = blocks[i].mean(axis=0)
        # stack
        xcorr = blocks[0].reshape(1, -1)
        for i in range(1, len(blocks)):
            xcorr = np.concatenate([xcorr, blocks[i].reshape(1, -1)], axis=0)
    else:
        xcorr = pairwise_xcorr_sorted
    return xcorr


def recall(memory, prednet, input_partial, mask, T=5):
    x = memory.recall_from_partial(input_partial.to(device), mask=mask)
    # get recalled dynamics
    rh_recalled = x[:, mask:]
    w_recalled = prednet.hypernet(rh_recalled)
    # predict seqeunce with r and rh
    recalled_sequence = torch.zeros(5, 324)
    r = input_partial[:, :mask].to(device)
    R = torch.zeros(T, r.size(1))
    for t in range(T):
        R[t] = r.clone().detach()
        img = prednet.spatial_decoder(r)
        recalled_sequence[t] = img.cpu().detach().reshape(-1)
        r = prednet.temporal_prediction_(r, rh_recalled).reshape(1, -1)
    return rh_recalled.clone().detach(), w_recalled.clone().detach(), R, recalled_sequence


def testing_recall(memory, prednet, R_conditioning, data_batch_start, save_dir):

    data_batch_mid = torch.zeros_like(data_batch_start)
    data_batch_end = torch.zeros_like(data_batch_start)
    data_batch_mid[:, :mask] = R_conditioning[2]
    data_batch_end[:, :mask] = R_conditioning[-1]
    # keep the content part
    data_corrupt_start = corrupt_input(data_batch_start, mask)
    data_corrupt_mid = corrupt_input(data_batch_mid, mask)
    data_corrupt_end = corrupt_input(data_batch_end, mask)
    # recall higher level rh and sequence
    rh_recalled_start, w_recalled_start, R_recalled_start, seq_recalled_start = recall(memory, prednet, data_corrupt_start, mask)
    rh_recalled_mid, w_recalled_mid, R_recalled_mid, seq_recalled_mid = recall(memory, prednet, data_corrupt_mid, mask)
    rh_recalled_end, w_recalled_end, R_recalled_end, seq_recalled_end = recall(memory, prednet, data_corrupt_end, mask)
    # conditioning
    rh_conditioning = data_batch_start[:, mask:].to(device)
    w_conditioning = prednet.hypernet(rh_conditioning).clone().detach()
    # save 
    np.save(op.join(save_dir, "rh_conditioning.npy"), rh_conditioning.cpu().detach().numpy())
    np.save(op.join(save_dir, "rh_recalled_start.npy"), rh_recalled_start.cpu().detach().numpy())
    np.save(op.join(save_dir, "rh_recalled_mid.npy"), rh_recalled_mid.cpu().detach().numpy())
    np.save(op.join(save_dir, "rh_recalled_end.npy"), rh_recalled_end.cpu().detach().numpy())
    np.save(op.join(save_dir, "w_conditioning.npy"), w_conditioning.cpu().detach().numpy())
    np.save(op.join(save_dir, "w_recalled_start.npy"), w_recalled_start.cpu().detach().numpy())
    np.save(op.join(save_dir, "w_recalled_mid.npy"), w_recalled_mid.cpu().detach().numpy())
    np.save(op.join(save_dir, "w_recalled_end.npy"), w_recalled_end.cpu().detach().numpy())
    np.save(op.join(save_dir, "R_conditioning.npy"), R_conditioning.cpu().detach().numpy())
    np.save(op.join(save_dir, "R_recalled_start.npy"), R_recalled_start.cpu().detach().numpy())
    np.save(op.join(save_dir, "R_recalled_mid.npy"), R_recalled_mid.cpu().detach().numpy())
    np.save(op.join(save_dir, "R_recalled_end.npy"), R_recalled_end.cpu().detach().numpy())
    np.save(op.join(save_dir, "seq_recalled_start.npy"), seq_recalled_start.cpu().detach().numpy())
    np.save(op.join(save_dir, "seq_recalled_mid.npy"), seq_recalled_mid.cpu().detach().numpy())
    np.save(op.join(save_dir, "seq_recalled_end.npy"), seq_recalled_end.cpu().detach().numpy())
    return R_recalled_start


def normalize(x):
    return (x - x.min()) / (x.max() - x.min())


def correlogram(prednet, peak_neurons, R_conditioning, R_recalled_start, R_recalled_start_raw, save_dir):

    U = prednet.spatial_decoder.weight.cpu().detach().numpy().T
    U = U[peak_neurons]

    xcorr_con  = xcorr(U, R_conditioning[:,peak_neurons].T, avg=False)[::-1]
    xcorr_start = xcorr(U, R_recalled_start[:,peak_neurons].T, avg=False)[::-1]
    xcorr_start_raw = xcorr(U, R_recalled_start_raw[:,peak_neurons].T, avg=False)[::-1]
    
    # normalize
    xcorr_con_norm = normalize(xcorr_con)
    xcorr_start_norm = normalize(xcorr_start)
    xcorr_start_raw_norm = normalize(xcorr_start_raw)
    xcorr_diff_norm = normalize(xcorr_start - xcorr_start_raw)
    # save 
    np.save(op.join(save_dir, "xcorr_con_norm.npy"), xcorr_con_norm)
    np.save(op.join(save_dir, "xcorr_start_norm.npy"), xcorr_start_norm)
    np.save(op.join(save_dir, "xcorr_start_raw_norm.npy"), xcorr_start_raw_norm)
    np.save(op.join(save_dir, "xcorr_diff_norm.npy"), xcorr_diff_norm)


if __name__ == '__main__':

    args = parser.parse_args()
    # load data and model
    fpath = args.model_dir
    data_dir = args.data_dir
    params = utils.Params(op.join(fpath, 'params.json'))
    params.cuda = torch.cuda.is_available()
    params.shuffle = False

    torch.manual_seed(200)
    
    dataloaders = data_loader.fetch_dataloader(['train'], data_dir, params, flag='memory')
    train_dl = dataloaders['train']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # memory network
    memory = Memory(params, device).to(device)
    raw_memory = copy.deepcopy(memory)    
    ckpt = utils.load_checkpoint(op.join(fpath, 'best.pth.tar'), memory)
    memory = memory.to(device)
    memory.eval()
    
    # predictive coding network
    prednet_fpath = "../experiments/mnist_ista/"
    prednet_params = utils.Params(op.join(prednet_fpath, 'params.json'))
    prednet = DynPredNet(prednet_params, device)
    ckpt = utils.load_checkpoint(op.join(prednet_fpath, 'best.pth.tar'), prednet)
    prednet = prednet.to(device)
    prednet.eval()

    # test recall
    mask = prednet.r_dim
    # conditioning data
    R_conditioning = torch.tensor(np.load("../data/memory/r_full_seq.npy"))
    peak_neurons = []
    for t in range(5):
        peak_neurons += list(np.argsort(R_conditioning.numpy()[t])[::-1][:1])
    data_batch_start = train_dl.dataset.data[:params.batch_size]
    
    save_dir = op.join("results", "recall", "trained")
    if not op.exists(save_dir):
        os.makedirs(save_dir)
    R_recalled_start = testing_recall(memory, prednet, R_conditioning, data_batch_start, save_dir)
    
    save_dir_raw = op.join("results", "recall", "raw")
    if not op.exists(save_dir_raw):
        os.makedirs(save_dir_raw)
    R_recalled_start_raw = testing_recall(raw_memory, prednet, R_conditioning, data_batch_start, save_dir_raw)

    correlogram(prednet, peak_neurons, R_conditioning, R_recalled_start, R_recalled_start_raw, save_dir)
