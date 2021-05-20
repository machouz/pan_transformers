import math
import random

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from termcolor import colored
from torch.utils.data import DataLoader, Dataset

import Input


def collate(batch):
    return [{
        'id': elem['id'],
        'input': elem['input'],
        'label': torch.tensor(elem['label'], dtype=torch.long)
    } for elem in batch]


class FeedData(pl.LightningDataModule):

    def __init__(self, path, batch_size: int = 1):
        super().__init__()
        self.data = Input.get_data_dict(path)
        random.shuffle(self.data)
        self.batch_size = batch_size

    def setup(self, stage=None):
        self._train, self._validate = np.split(
            self.data, [int(.8*len(self.data))])

    def train_dataloader(self):
        return DataLoader(self._train, collate_fn=collate, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self._validate, collate_fn=collate, batch_size=self.batch_size)
