import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import softmax
from termcolor import colored
from transformers import AutoModel, AutoTokenizer

from Model.DefaultModel import DefaultModel


class CustomHate(DefaultModel):
    def __init__(self, pretrained_model_name):
        super(CustomHate, self).__init__()

        self.pretrained_model = AutoModel.from_pretrained(
            pretrained_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, normalization=True)
        self.pretrained_model.resize_token_embeddings(len(self.tokenizer))

        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        self.linear1 = nn.Sequential(  # Sequential,
            nn.Linear(self.pretrained_model.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(p=0.8),
        )
        self.linear2 = nn.Sequential(  # Sequential,
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )
        self.linear3 = nn.Sequential(  # Sequential,
            nn.Linear(64, 2),
        )

    def forward(self, input):
        encoded_input = self.tokenize(input)
        output = self.pretrained_model(**encoded_input, return_dict=True)
        output = output.last_hidden_state

        output = output[:, 0, :]
        # extract the 1st token's embeddings
        output = self.linear1(output)
        output = self.linear2(output)

        output = output.unsqueeze(0)

        output = output.mean(1)
        output = self.linear3(output)

        return F.log_softmax(output)
