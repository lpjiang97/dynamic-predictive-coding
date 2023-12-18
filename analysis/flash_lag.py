import argparse
import os
import os.path as op
import sys
sys.path.insert(0, op.abspath("../."))

import numpy as np
from scipy import special
from scipy.ndimage import center_of_mass
import torch

import utils
from models.predictive_coding_ista import DynPredNet
from evaluation import record


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='../data/postdiction', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='../experiments/mnist_ista', help="Directory containing params.json")
parser.add_argument('--step', default=3, type=int, help='number of steps to predict ahead')


def predict_ahead(model, r, r2, T):
    batch_size = r.shape[0]
    recalled_sequence = torch.zeros(batch_size, T, 324)
    for t in range(T):
        r = model.temporal_prediction_(r, r2)
        img = model.spatial_decoder(r) 
        recalled_sequence[:,t] = img.cpu().detach()
    return recalled_sequence[:,T-1,:]


def compute_original_location(x):
    batch_size = x.shape[0] 
    loc = np.zeros((batch_size, 2))
    for b in range(batch_size):
        loc[b] = center_of_mass(x[b].reshape(18,18))
    return loc


def compute_interference_loc(model, r_p, R2, T):
    batch_size = r_p.shape[0]
    converge_time = R2.shape[0] 
    loc = np.zeros((converge_time, batch_size, 2))
    for t in range(converge_time):
        predicted_x = predict_ahead(model, r_p, R2[t], T).cpu().detach().numpy()
        for b in range(batch_size):
            loc[t, b] = center_of_mass(predicted_x[b].reshape(18,18))
    return np.swapaxes(loc, 0, 1)


def compute_loc_diff(original_loc, interference_loc, motion):
    batch_size = original_loc.shape[0] 
    converge_time = interference_loc.shape[1]
    location_diff = np.zeros((batch_size, converge_time))
    for b in range(batch_size):
        # go through time
        for t in range(converge_time):
            # going to the right, originally
            if motion[b,5] == 3:
                location_diff[b,t] = interference_loc[b,t,1] - original_loc[b,1]
            elif motion[b,5] == 1:
                location_diff[b,t] = original_loc[b,1] - interference_loc[b,t,1]
            else:
                raise ValueError("Motion not supported") 
    return location_diff


def compute_flash_lag(moving_loc, flash_loc, motion):
    batch_size = moving_loc.shape[0] 
    location_diff = np.zeros((batch_size))
    for b in range(batch_size):
        # going to the right, originally
        if motion[b,5] == 3:
            location_diff[b] = moving_loc[b,-1,1] - flash_loc[b,-1,1]
        elif motion[b,5] == 1:
            location_diff[b] = flash_loc[b,-1,1] - moving_loc[b,-1,1] 
        else:
            raise ValueError("Motion not supported") 
    return location_diff


def test_and_save(model, X, T, original_batch_size, motion, filename, start_mode, test_mode, flash_loc=None):

    # compute original results
    result_dict = record(model, X, input_dim=params.input_dim, turnoff=10)
    
    r_p = result_dict["R_hat"][:, 5, :].to(device)
    r2 = result_dict["R2_hat"][:, 5, :].to(device)

    # different modes
    if start_mode == "cold":
        r2 = torch.zeros_like(r2)
    elif start_mode == "continuous":
        pass
    else:
        raise ValueError("start mode not supported")    

    if test_mode == "terminate":
        x = torch.zeros_like(result_dict["I"][:, 4, :].to(device))
    elif test_mode == "stopped":
        x = result_dict["I"][:, 5, :].to(device)
    elif test_mode == 'reversal':
        x = result_dict["I"][:, 4, :].to(device)
    elif test_mode == 'continuous': 
        x = result_dict["I"][:, 6, :].to(device)
    else:
        raise ValueError("test mode not supported")

    _, _, _, R, R2 = model.inf(x, r_p, r2, record=True)
    # compute original location
    original_loc = compute_original_location(result_dict["I_hat"][:,5,:].numpy())
    # compute interference location
    interference_loc = compute_interference_loc(model, r_p, R2, T)
    original_loc = original_loc[:original_batch_size] 
    interference_loc = interference_loc[:original_batch_size]
    # save
    save_dir = op.join("results", "interference")
    if not op.exists(save_dir):
        os.makedirs(save_dir)
    # compute diff
    if flash_loc is None:
        diff = compute_loc_diff(original_loc, interference_loc, motion)
        np.save(op.join(save_dir, filename + "_diff.npy"), diff)
    else:
        flash_lag = compute_flash_lag(interference_loc, flash_loc, motion) 
        np.save(op.join(save_dir, filename + "_flash_lag.npy"), flash_lag)
    return interference_loc


if __name__ == '__main__':
    
    args = parser.parse_args()
    # load data and model
    fpath = args.model_dir
    data_dir = args.data_dir 
    T = args.step

    params = utils.Params(op.join(fpath, 'params.json'))
    params.cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if params.cuda else "cpu")
    params.shuffle = False 

    model = DynPredNet(params, device)
    ckpt = utils.load_checkpoint(op.join(fpath, 'last.pth.tar'), model)
    model = model.to(device)
    model.eval()

    # load data 
    X = torch.tensor(np.load(op.join(data_dir, 'interference_data.npy')), dtype=torch.float32)
    original_batch_size = X.size(0)
    X = X.repeat(5, 1, 1, 1)
    X = X.reshape(X.size(0), X.size(1), -1).to(device)
    motion = np.load(op.join(data_dir, 'interference_motion.npy'))
    
    # test different conditions    
    test_and_save(model, X, T, original_batch_size, motion, f"cold_reversal_T-{T}", "cold", "reversal")
    # cold - terminate == flash
    flash_loc = test_and_save(model, X, T, original_batch_size, motion, f"cold_terminate_T-{T}", "cold", "terminate")
    test_and_save(model, X, T, original_batch_size, motion, f"cold_stopped_T-{T}", "cold", "stopped")
    test_and_save(model, X, T, original_batch_size, motion, f"cold_continuous_T-{T}", "cold", "continuous")

    test_and_save(model, X, T, original_batch_size, motion, f"continuous_reversal_T-{T}", "continuous", "reversal")
    test_and_save(model, X, T, original_batch_size, motion, f"continuous_terminate_T-{T}", "continuous", "terminate")
    test_and_save(model, X, T, original_batch_size, motion, f"continuous_stopped_T-{T}", "continuous", "stopped")
    test_and_save(model, X, T, original_batch_size, motion, f"continuous_continuous_T-{T}", "continuous", "continuous")

    # compute flash lag
    test_and_save(model, X, T, original_batch_size, motion, f"cold_reversal_T-{T}", "cold", "reversal", flash_loc=flash_loc)
    # cold - terminate == flash
    flash_loc = test_and_save(model, X, T, original_batch_size, motion, f"cold_terminate_T-{T}", "cold", "terminate", flash_loc=flash_loc)
    test_and_save(model, X, T, original_batch_size, motion, f"cold_stopped_T-{T}", "cold", "stopped", flash_loc=flash_loc)
    test_and_save(model, X, T, original_batch_size, motion, f"cold_continuous_T-{T}", "cold", "continuous", flash_loc=flash_loc)

    test_and_save(model, X, T, original_batch_size, motion, f"continuous_reversal_T-{T}", "continuous", "reversal", flash_loc=flash_loc)
    test_and_save(model, X, T, original_batch_size, motion, f"continuous_terminate_T-{T}", "continuous", "terminate", flash_loc=flash_loc)
    test_and_save(model, X, T, original_batch_size, motion, f"continuous_stopped_T-{T}", "continuous", "stopped", flash_loc=flash_loc)
    test_and_save(model, X, T, original_batch_size, motion, f"continuous_continuous_T-{T}", "continuous", "continuous", flash_loc=flash_loc)

 