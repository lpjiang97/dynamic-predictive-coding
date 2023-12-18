import argparse
import logging
import os.path as op
import sys
sys.path.insert(0, op.abspath(op.join(op.dirname(__file__), '.')))

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
#from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import utils
#from evaluation import evaluate
from models.memory import Memory
import models.data_loader as data_loader


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/memory',
                    help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'


def train(model, optimizer, dataloader, params):

    # set model to training mode
    model.train()
    optimizer.zero_grad()
    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = utils.RunningAverage()
    # take one example to over-train
    data_batch = next(iter(dataloader))
    # move to GPU if available
    if params.cuda:
        data_batch = data_batch.cuda(non_blocking=True)

    r, _ = model(data_batch)
    loss = torch.pow(data_batch - model.predict(r), 2).sum(1).mean(0) + torch.pow(r - model.b, 2).sum(1).mean(0)
    loss_dict = {"loss": loss.item()}
    loss.backward()
    optimizer.step()
    # compute all metrics on this batch
    summ.append(loss_dict)
    # update the average loss
    loss_avg.update(loss.item())

    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric]
                                     for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)

    return metrics_mean


def train_and_evaluate(model, dataloader, optimizer, scheduler, params, writer, model_dir, restore_file=None):

    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = op.join(args.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

    best_val_loss = float("inf")

    for epoch in range(params.num_epochs):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # Train model
        train_metrics = train(model, optimizer, dataloader, params)
        # Evaluate for one epoch on validation set
        # write to tensorboard
        writer.add_scalar("Loss", train_metrics['loss'], epoch)

        val_loss = train_metrics['loss']
        is_best = val_loss <= best_val_loss

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                              is_best=is_best,
                              checkpoint=model_dir)

        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best loss")
            best_val_loss = val_loss
            # Save best val metrics in a json file in the model directory
            best_json_path = op.join(model_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(train_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = op.join(model_dir, "metrics_val_last_weights.json")
        utils.save_dict_to_json(train_metrics, last_json_path)

        # adjust learning rate
        scheduler.step()
    

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
    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)

    # create writer
    writer = SummaryWriter(log_dir=op.join(fpath, 'tensorboard', 'train'))

    # Set the logger
    utils.set_logger(op.join(fpath, 'train.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    # fetch dataloaders
    params.shuffle = False
    dataloaders = data_loader.fetch_dataloader(['train'], args.data_dir, params, flag='memory')
    dl = dataloaders['train']

    logging.info("- done.")

    device = torch.device("cuda:0" if params.cuda else "cpu")

    # load predictive coding model
    # prednet = DynPredNet(params, device)
    # ckpt = utils.load_checkpoint(op.join(fpath, 'prednet.pth.tar'), prednet)
    # prednet = prednet.to(device)
    # prednet.eval()

    # Define the model and optimizer
    model = Memory(params, device).to(device)
    optimizer = optim.Adam(model.parameters(), params.mem_learning_rate)

    #scheduler = ExponentialLR(optimizer, gamma=params.learning_rate_gamma)
    scheduler = ExponentialLR(optimizer, gamma=params.learning_rate_gamma)

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(model, dl, optimizer, scheduler, params, writer, args.model_dir, args.restore_file)

