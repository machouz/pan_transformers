import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import softmax
from termcolor import colored
from transformers import AutoModel, AutoTokenizer

from Model.DefaultModel import DefaultModel


class ConvolutionalHate(DefaultModel):
    def __init__(self, pretrained_model_name):
        super(ConvolutionalHate, self).__init__()

        self.pretrained_model = AutoModel.from_pretrained(
            pretrained_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, normalization=True)
        self.pretrained_model.resize_token_embeddings(len(self.tokenizer))

        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        # -------Sentence model-------
        # OUTPUT 200 768

        # -------Feed model-------
        # Conv output   = [(W−K+2P)/S]+1
        #               => W:input, K: kernel, P: padding, S: stride
        self.conv1 = nn.Sequential(
            # (200-5+2*2)/5+1=40 (768-5+2*2)/5+1=154
            nn.Conv2d(1, 1, kernel_size=5, stride=5, padding=2),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            #  (40-5)/10+1=8 (154-5)/10+1=30
            nn.MaxPool2d(kernel_size=5, stride=10)
        )

        # OUTPUT 5 200/8=50 768/8=16
        self.linear3 = nn.Sequential(  # Sequential,
            nn.Linear(1 * 4 * 15, 12),
            nn.Dropout(p=0.3),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(  # Sequential,
            nn.Linear(12, 2),
        )

    def forward(self, input):
        encoded_input = self.tokenize(input)
        output = self.pretrained_model(**encoded_input, return_dict=True)
        output = output.last_hidden_state

        output = output[:, 0, :]

        output = output.unsqueeze(0).unsqueeze(0)

        output = self.conv1(output)

        output = output.reshape(output.size(0), -1)

        output = self.linear3(output)
        output = self.classifier(output)

        return F.log_softmax(output)
