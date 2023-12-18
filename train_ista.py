import argparse
import logging
import os.path as op
import sys
sys.path.insert(0, op.abspath(op.join(op.dirname(__file__), '.')))

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import utils
from evaluation import evaluate
from models.predictive_coding_single import DynPredNet as SingleNet
from models.predictive_coding_ista import DynPredNet
import models.data_loader as data_loader



parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/forest',
                    help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'


def train(model, optimizer, dataloader, params):
    """Train the model on `num_steps` batches

    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to training mode
    model.train()

    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = utils.RunningAverage()

    # Use tqdm for progress bar
    with tqdm(total=len(dataloader), dynamic_ncols=True) as t:
        for i, train_batch in enumerate(dataloader):
            # move to GPU if available
            if params.cuda:
                train_batch = train_batch.cuda(non_blocking=True)

            spatial_loss, temp_loss, r2_losses, _, _ = model(train_batch)
            # compute loss
            loss_dict = {
                "spatial_loss": spatial_loss.item(),
                "temp_loss": temp_loss.item(),
            }
            # clear previous gradients, compute gradients of all variables wrt loss
            for opt in optimizer: opt.zero_grad()
            loss = spatial_loss + temp_loss
            loss.backward()
            # performs updates using calculated gradients
            for opt in optimizer: opt.step()
            # normalize
            model.normalize()
            if i % params.save_summary_steps == 0:
                # compute all metrics on this batch
                summ.append(loss_dict)

            # update the average loss
            loss_avg.update(loss.item())

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()

    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric]
                                     for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)

    return metrics_mean, r2_losses


def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, scheduler, params, train_writer, test_writer, model_dir,
                       restore_file=None, two_level=True):
    """Train the model and evaluate every epoch.

    Args:
        model: (torch.nn.Module) the neural network
        train_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        val_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches validation data
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
        restore_file: (string) optional- name of file to restore from (without its extension .pth.tar)
    """
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = op.join(args.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)
    E = params.num_epochs
    best_val_loss = float("inf")
    test_losses = np.zeros(E)

    for epoch in range(E):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # Train model
        train_metrics, r2_losses_train = train(model, optimizer, train_dataloader, params)
        # Evaluate for one epoch on validation set
        test_metrics, r2_losses_test, test_loss = evaluate(model, val_dataloader, params)
        test_losses[epoch] = test_loss
        
        # write to tensorboard
        train_writer.add_scalar("Total", train_metrics['spatial_loss'] + train_metrics['temp_loss'], epoch)
        train_writer.add_scalar("Spatial Loss", train_metrics['spatial_loss'], epoch)
        train_writer.add_scalar("Temporal Loss", train_metrics['temp_loss'], epoch)

        fig, ax = utils.plot_spatial_rf(model.spatial_decoder.weight.T.data.reshape(model.r_dim, -1).detach().cpu().numpy()[:100])
        train_writer.add_figure("RF", fig, epoch)

        test_writer.add_scalar("Total", test_metrics['spatial_loss'] + test_metrics['temp_loss'], epoch)
        test_writer.add_scalar("Spatial Loss", test_metrics['spatial_loss'], epoch)
        test_writer.add_scalar("Temporal Loss", test_metrics['temp_loss'], epoch)

        if two_level:
            fig = utils.plot_r2_loss(r2_losses_train)
            train_writer.add_figure("R2 loss", fig, global_step=epoch)
            fig = utils.plot_r2_loss(r2_losses_test)
            test_writer.add_figure("R2 loss", fig, global_step=epoch)

        val_loss = test_metrics['spatial_loss'] + test_metrics['temp_loss']
        is_best = val_loss <= best_val_loss

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict': [opt.state_dict() for opt in optimizer]},
                              is_best=is_best,
                              checkpoint=model_dir)
        if epoch % 10 == 9:
            utils.save_checkpoint({'epoch': epoch + 1,
                                   'state_dict': model.state_dict(),
                                   'optim_dict': [opt.state_dict() for opt in optimizer]},
                                  is_best=is_best,
                                  checkpoint=model_dir,
                                  filename=f'model_epoch_{epoch+1}.pth.tar')

        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best loss")
            best_val_loss = val_loss

            # Save best val metrics in a json file in the model directory
            best_json_path = op.join(model_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(test_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = op.join(model_dir, "metrics_val_last_weights.json")
        utils.save_dict_to_json(test_metrics, last_json_path)

        # adjust learning rate
        if epoch < 100:
            for sched in scheduler: sched.step()
    
    # save test losses
    np.save(op.join(model_dir, 'test_losses.npy'), test_losses)


if __name__ == '__main__':

    # Load the parameters from json file
    args = parser.parse_args()
    fpath = args.model_dir
    json_path = op.join(fpath, 'params.json')
    assert op.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    utils.set_seed(params.seed)
    
    # create writer
    train_writer = SummaryWriter(log_dir=op.join(fpath, 'tensorboard', 'train'))
    test_writer = SummaryWriter(log_dir=op.join(fpath, 'tensorboard', 'test'))

    # Set the logger
    utils.set_logger(op.join(fpath, 'train.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    # fetch dataloaders
    params.shuffle = True
    dataloaders = data_loader.fetch_dataloader(
        ['train', 'test'], args.data_dir, params)
    train_dl = dataloaders['train']
    val_dl = dataloaders['test']

    logging.info("- done.")

    device = torch.device("cuda:0" if params.cuda else "cpu")
    # Define the model and optimizer
    two_level = params.model == "two"

    if two_level:
        model = DynPredNet(params, device).to(device)
        optimizer = [
            optim.SGD(model.spatial_decoder.parameters(), params.learning_rate_s),
            optim.Adam([model.temporal], params.learning_rate_t),
            optim.Adam(model.hypernet.parameters(), params.learning_rate_t)
        ]
    else:
        model = SingleNet(params, device).to(device)
        optimizer = [
            optim.SGD(model.spatial_decoder.parameters(), params.learning_rate_s),
            optim.Adam(model.temporal.parameters(), params.learning_rate_t),
        ] 

    scheduler = [ExponentialLR(optimizer[0], gamma=params.learning_rate_gamma)] + \
        [ExponentialLR(opt, gamma=params.learning_rate_gamma-0.03) for opt in optimizer[1:]]

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(model, train_dl, val_dl, optimizer, scheduler, params, train_writer, 
                       test_writer, args.model_dir, args.restore_file, two_level=two_level)

