import argparse
import logging
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/64x64_SIGNS',
                    help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")


def evaluate(model, dataloader, params):
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to evaluation mode
    model.eval()

    # summary for current eval loop
    summ = []
    num_sample = 0
    test_loss = 0
    # compute metrics over the dataset
    for data_batch in dataloader:
        batch_size = data_batch.size(0)
        # move to GPU if available
        if params.cuda:
            data_batch = data_batch.cuda(non_blocking=True)

        spatial_loss, temp_loss, r2_losses, _, _ = model(data_batch)
        num_sample += batch_size
        test_loss += spatial_loss.item() * batch_size + temp_loss.item() * batch_size

        # compute loss
        loss_dict = {
            "spatial_loss": spatial_loss.item(),
            "temp_loss": temp_loss.item(),
        }
        # compute all metrics on this batch
        summ.append(loss_dict)

    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric]
                                     for x in summ]) for metric in summ[0]}
    # compute per sample metric 
    test_loss /= num_sample

    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)
    return metrics_mean, r2_losses, test_loss


def record_test_set(params, model, data_loader, input_dim=256, mixture=True, turnoff=None):
    model.eval()
    result_dict = None
    for data_batch in data_loader:
        if params.cuda:
            data_batch = data_batch.cuda(non_blocking=True)
        r_d = record(model, data_batch, input_dim=input_dim, mixture=mixture, turnoff=turnoff)
        if result_dict is None:
            result_dict = r_d
        else:
            for key in result_dict:
                result_dict[key] = torch.cat((result_dict[key], r_d[key]), axis=0)
    return result_dict    


def record(model, data_batch, input_dim=256, mixture=True, turnoff=None):
    model.eval()
    batch_size = data_batch.size(0)
    T = data_batch.size(1)
    if turnoff is None:
        turnoff = T
    assert turnoff <= T, "Input turnoff time larger than the total sequence length"
    # saving values
    I_bar = torch.zeros((batch_size, T, input_dim))       # Image prediction from hypernet
    I_hat = torch.zeros((batch_size, T, input_dim))       # Image correction (turned off after turnoff)
    I = torch.zeros((batch_size, T, input_dim))           # True input
    R_bar = torch.zeros((batch_size, T, model.r_dim))     # prediction from hypernet
    R_hat = torch.zeros((batch_size, T, model.r_dim))     # ISTA correction (turned off after turnoff)
    R2_hat = torch.zeros((batch_size, T, model.r2_dim))   # Embedding (same after turnoff)
    if mixture:
        W = torch.zeros((batch_size, T, model.mix_dim))       # Mixture weights
    # initialize embedding
    r, r2 = model.init_code_(batch_size)
    R_bar[:, 0] = r.clone().detach().cpu()
    R2_hat[:, 0] = r2.clone().detach().cpu()
    I_bar[:, 0] = model.spatial_decoder(r).clone().detach().cpu()
    # p(r_1 | I_1)
    r = model.inf_first_step(data_batch[:, 0])
    R_hat[:, 0] = r.clone().detach().cpu()
    I_hat[:, 0] = model.spatial_decoder(r).clone().detach().cpu()
    I[:, 0] = data_batch[:, 0]
    # input on
    for t in range(1, turnoff):
        r_p = r.clone().detach()
        # hypernet prediction
        r_bar = model.temporal_prediction_(r_p, r2)
        R_bar[:, t] = r_bar.clone().detach().cpu()
        I_bar[:, t] = model.spatial_decoder(r_bar).clone().detach().cpu()
        # inference
        r, r2, _ = model.inf(data_batch[:, t], r_p, r2.clone().detach())
        R_hat[:, t] = r.clone().detach().cpu()
        I_hat[:, t] = model.spatial_decoder(r).clone().detach().cpu()
        I[:, t] = data_batch[:, t]
        R2_hat[:, t]= r2.clone().detach().cpu()
        if mixture:
            w = model.hypernet(r2)
            W[:, t] = w.reshape(batch_size, -1).clone().detach().cpu()
    # input off, no more inference
    for t in range(turnoff, T):
        # predict
        r_bar = model.temporal_prediction_(r, r2)
        #r_hat = model.prediction_(r, r2)
        R_bar[:, t] = r_bar.clone().detach().cpu()
        I_bar[:, t] = model.spatial_decoder(r_bar).clone().detach().cpu()
        # no more correction
        R_hat[:, t] = R_bar[:, t]
        I_hat[:, t] = I_bar[:, t]
        # no more fitting embedding
        R2_hat[:, t] = R2_hat[:, t-1]
        I[:, t] = data_batch[:, t]
        if mixture:
            W[:, t] = W[:, t-1]
        r = r_bar
    # result dict
    result_dict = {
        "I_bar": I_bar,
        "I_hat": I_hat,
        "I": I,
        "R_bar": R_bar,
        "R_hat": R_hat,
        "R2_hat": R2_hat,
    }
    if mixture:
        result_dict["W"] = W
    return result_dict


def record_two_trans(model, data_batch, input_dim=256, turnoff=None):
    model.eval()
    batch_size = data_batch.size(0)
    T = data_batch.size(1)
    if turnoff is None:
        turnoff = T
    assert turnoff <= T, "Input turnoff time larger than the total sequence length"
    # saving values
    I_bar = torch.zeros((batch_size, T, input_dim))
    I_hat = torch.zeros((batch_size, T, input_dim))
    I = torch.zeros((batch_size, T, input_dim))
    R_bar = torch.zeros((batch_size, T, model.r_dim))
    R_hat = torch.zeros((batch_size, T, model.r_dim))
    R2_bar = torch.zeros((batch_size, T, model.r2_dim))
    R2_hat = torch.zeros((batch_size, T, model.r2_dim))
    W_bar = torch.zeros((batch_size, T, model.mix_dim))      
    W_hat = torch.zeros((batch_size, T, model.mix_dim))
    # initialize embedding
    r, r2 = model.init_code_(batch_size)
    R_bar[:, 0] = r.clone().detach().cpu()
    R2_hat[:, 0] = r2.clone().detach().cpu()
    I_bar[:, 0] = model.spatial_decoder(r).clone().detach().cpu()
    # p(r_1 | I_1)
    r = model.inf_first_step(data_batch[:, 0])
    R_hat[:, 0] = r.clone().detach().cpu()
    I_hat[:, 0] = model.spatial_decoder(r).clone().detach().cpu()
    I[:, 0] = data_batch[:, 0]
    r_p = r.clone().detach()
    # second step 
    r_bar = model.temporal_prediction_(r_p, r2)
    R_bar[:, 1] = r_bar.clone().detach().cpu()
    I_bar[:, 1] = model.spatial_decoder(r_bar).clone().detach().cpu()
    # inference
    r, r2, _ = model.inf(data_batch[:, 1], r.clone().detach())         
    R_hat[:, 1] = r.clone().detach().cpu()
    I_hat[:, 1] = model.spatial_decoder(r).clone().detach().cpu()
    I[:, 1] = data_batch[:, 1]
    R2_hat[:, 1]= r2.clone().detach().cpu()
    w = model.hypernet(r2)
    W_hat[:, 1] = w.reshape(batch_size, -1).clone().detach().cpu()
    r_p = r.clone().detach()
    r2_p = r2.clone().detach()

    # input on
    for t in range(2, turnoff):
        # predictions
        r2_bar = model.temporal2(torch.cat([r_p, r2_p], dim=1)) 
        R2_bar[:, t] = r2_bar.clone().detach().cpu()
        w = model.hypernet(r2_bar)
        W_bar[:, t] = w.reshape(batch_size, -1).clone().detach().cpu()
        r_bar = model.temporal_prediction_(r_p, r2_bar) 
        R_bar[:, t] = r_bar.clone().detach().cpu()
        I_bar[:, t] = model.spatial_decoder(r_bar).clone().detach().cpu()
        # inference
        r, r2, _ = model.inf(data_batch[:, t], r_p, r2_p=r2_p)
        w = model.hypernet(r2)
        W_hat[:, t] = w.reshape(batch_size, -1).clone().detach().cpu()
        R2_hat[:, t] = r2.clone().detach().cpu()
        R_hat[:, t] = r.clone().detach().cpu()
        I_hat[:, t] = model.spatial_decoder(r).clone().detach().cpu()
        I[:, t] = data_batch[:, t]
        R2_hat[:, t]= r2.clone().detach().cpu()
        # move previous codes
        r_p = r.clone().detach()
        r2_p = r2.clone().detach()
    # input off
    for t in range(turnoff, T):
        # predictions
        r2_bar = model.temporal2(torch.cat([r_p, r2_p], dim=1)) 
        R2_bar[:, t] = r2_bar.clone().detach().cpu()
        w = model.hypernet(r2_bar)
        W_bar[:, t] = w.reshape(batch_size, -1).clone().detach().cpu()
        r_bar = model.temporal_prediction_(r_p, r2_bar) 
        R_bar[:, t] = r_bar.clone().detach().cpu()
        I_bar[:, t] = model.spatial_decoder(r_bar).clone().detach().cpu()
        # no more corrections
        r = r_bar.clone().detach()
        r2 = r2_bar.clone().detach()
        # save
        R_hat[:, t] = R_bar[:, t]
        I_hat[:, t] = I_bar[:, t]
        R2_hat[:, t] = R2_bar[:, t]
        W_hat[:, t] = W_bar[:, t]
        I[:, t] = data_batch[:, t]
        # move previous codes
        r_p = r.clone().detach()
        r2_p = r2.clone().detach()
    # result dict
    result_dict = {
        "I_bar": I_bar,
        "I_hat": I_hat,
        "I": I,
        "R_bar": R_bar,
        "R_hat": R_hat,
        "R2_bar": R2_bar,
        "R2_hat": R2_hat,
        "W_bar": W_bar,
        "W_hat": W_hat
    }
    return result_dict


def record_three(model, data_batch, input_dim=256, turnoff=None):
    model.eval()
    batch_size = data_batch.size(0)
    T = data_batch.size(1)
    if turnoff is None:
        turnoff = T
    assert turnoff <= T, "Input turnoff time larger than the total sequence length"
    # saving values
    I_bar = torch.zeros((batch_size, T, input_dim))
    I_hat = torch.zeros((batch_size, T, input_dim))
    I = torch.zeros((batch_size, T, input_dim))
    R_bar = torch.zeros((batch_size, T, model.r_dim))
    R_hat = torch.zeros((batch_size, T, model.r_dim))
    R2_bar = torch.zeros((batch_size, T, model.r2_dim))
    R2_hat = torch.zeros((batch_size, T, model.r2_dim))
    W_bar = torch.zeros((batch_size, T, model.mix_dim))      
    W_hat = torch.zeros((batch_size, T, model.mix_dim))
    R3_hat = torch.zeros((batch_size, T, model.r3_dim))
    W2_hat = torch.zeros((batch_size, T, model.mix_dim_2))

    # initialize embedding
    r, r2, r3 = model.init_code_(batch_size)
    R_bar[:, 0] = r.clone().detach().cpu()
    R2_hat[:, 0] = r2.clone().detach().cpu()
    I_bar[:, 0] = model.spatial_decoder(r).clone().detach().cpu()
    # p(r_1 | I_1)
    r = model.inf_first_step(data_batch[:, 0])
    R_hat[:, 0] = r.clone().detach().cpu()
    I_hat[:, 0] = model.spatial_decoder(r).clone().detach().cpu()
    I[:, 0] = data_batch[:, 0]
    r_p = r.clone().detach()
    # second step 
    r_bar = model.temporal_prediction_one_(r_p, r2)
    R_bar[:, 1] = r_bar.clone().detach().cpu()
    I_bar[:, 1] = model.spatial_decoder(r_bar).clone().detach().cpu()
    # inference
    r, r2, _, _ = model.inf(data_batch[:, 1], r.clone().detach())         
    R_hat[:, 1] = r.clone().detach().cpu()
    I_hat[:, 1] = model.spatial_decoder(r).clone().detach().cpu()
    I[:, 1] = data_batch[:, 1]
    R2_hat[:, 1]= r2.clone().detach().cpu()
    w = model.hypernet(r2)
    W_hat[:, 1] = w.reshape(batch_size, -1).clone().detach().cpu()
    r_p = r.clone().detach()
    r2_p = r2.clone().detach()

    # input on
    for t in range(2, turnoff):
        # predictions
        r2_bar = model.temporal_prediction_two_(r_p, r2_p, r3) 
        R2_bar[:, t] = r2_bar.clone().detach().cpu()
        w = model.hypernet(r2_bar)
        W_bar[:, t] = w.reshape(batch_size, -1).clone().detach().cpu()
        r_bar = model.temporal_prediction_one_(r_p, r2_bar) 
        R_bar[:, t] = r_bar.clone().detach().cpu()
        I_bar[:, t] = model.spatial_decoder(r_bar).clone().detach().cpu()
        # inference
        r, r2, r3, _ = model.inf(data_batch[:, t], r_p, r2_p=r2_p, r3=r3.clone().detach())
        w2 = model.hypernet2(r3)
        W2_hat[:, t] = w2.reshape(batch_size, -1).clone().detach().cpu()
        w = model.hypernet(r2)
        W_hat[:, t] = w.reshape(batch_size, -1).clone().detach().cpu()
        R3_hat[:, t] = r3.clone().detach().cpu()
        R2_hat[:, t] = r2.clone().detach().cpu()
        R_hat[:, t] = r.clone().detach().cpu()
        I_hat[:, t] = model.spatial_decoder(r).clone().detach().cpu()
        I[:, t] = data_batch[:, t]
        R2_hat[:, t]= r2.clone().detach().cpu()
        # move previous codes
        r_p = r.clone().detach()
        r2_p = r2.clone().detach()
    # input off
    for t in range(turnoff, T):
        # predictions
        r2_bar = model.temporal_prediction_two_(r_p, r2_p, r3) 
        R2_bar[:, t] = r2_bar.clone().detach().cpu()
        w = model.hypernet(r2_bar)
        W_bar[:, t] = w.reshape(batch_size, -1).clone().detach().cpu()
        r_bar = model.temporal_prediction_one_(r_p, r2_bar) 
        R_bar[:, t] = r_bar.clone().detach().cpu()
        I_bar[:, t] = model.spatial_decoder(r_bar).clone().detach().cpu()
        # no more corrections
        r = r_bar.clone().detach()
        r2 = r2_bar.clone().detach()
        # save
        R3_hat[:, t] = R3_hat[:, t-1] 
        W2_hat[:, t] = W2_hat[:, t-1]
        R2_hat[:, t] = R2_bar[:, t]
        R_hat[:, t] = R_bar[:, t]
        I_hat[:, t] = I_bar[:, t]
        R2_hat[:, t] = R2_bar[:, t]
        W_hat[:, t] = W_bar[:, t]
        I[:, t] = data_batch[:, t]
        # move previous codes
        r_p = r.clone().detach()
        r2_p = r2.clone().detach()
    # result dict
    result_dict = {
        "I_bar": I_bar,
        "I_hat": I_hat,
        "I": I,
        "R_bar": R_bar,
        "R_hat": R_hat,
        "R2_bar": R2_bar,
        "R2_hat": R2_hat,
        "R3_hat": R3_hat,
        "W_bar": W_bar,
        "W_hat": W_hat,
        "W2_hat": W2_hat
    }
    return result_dict
