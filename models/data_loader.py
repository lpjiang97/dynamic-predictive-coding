import os.path as op
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class VideoDataset(Dataset):

    def __init__(self, data_path):
        """
        Args:
            data_path: (string) path to the data file
            ttype_path: (string) path to the trial type file
            transform
        """
        super(VideoDataset, self).__init__()
        d = np.load(data_path)
        d = d.reshape(d.shape[0], d.shape[1], -1)
        # split time in half: use length 10 sequences
        self.data = torch.tensor(d).float()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]


class MemoryDataset(Dataset):

    def __init__(self, data_path):
        """
        Args:
            data_path: (string) path to the data file
            ttype_path: (string) path to the trial type file
            transform
        """
        super(MemoryDataset, self).__init__()
        d = np.load(data_path)
        if len(d.shape) == 1:
            d = d.reshape(1, d.shape[0])
        # split time in half: use length 10 sequences
        self.data = torch.tensor(d).float()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]



def fetch_dataloader(types, data_dir, params, flag='video'):
    """
    Fetches the DataLoader object for each type in types from data_dir.

    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        data_dir: (string) directory containing the dataset
        params: (Params) hyperparameters

    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    dataloaders = {}

    for split in ['train', 'val', 'test']:
        if split in types:
            data_path = op.join(data_dir, f"data_{split}.npy")
            # use the train_transformer if training data, else use eval_transformer without random flip
            if flag == 'video':
                dl = DataLoader(VideoDataset(data_path), batch_size=params.batch_size, shuffle=params.shuffle,
                                num_workers=params.num_workers, pin_memory=params.cuda)
            elif flag == 'memory':
                dl = DataLoader(MemoryDataset(data_path), batch_size=params.batch_size, shuffle=params.shuffle,
                                num_workers=params.num_workers, pin_memory=params.cuda)
            else:
                raise NotImplementedError
            dataloaders[split] = dl

    return dataloaders
