import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import softmax
from termcolor import colored
from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM


from Model.DefaultModel import DefaultModel


class LSTMHate(DefaultModel):
    def __init__(self, pretrained_model_name):
        super(LSTMHate, self).__init__()
        self.pretrained_model = AutoModel.from_pretrained(
            pretrained_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, normalization=True)

        self.pretrained_model.resize_token_embeddings(len(self.tokenizer))

        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        # -------Sentence model-------

        # -------Feed model-------
        self.lstm = nn.LSTM(input_size=768,
                            hidden_size=32,
                            num_layers=2,
                            dropout=0.5,
                            batch_first=True,
                            bidirectional=True)
        self.classifier = nn.Sequential(  # Sequential,
            nn.Dropout(0.5),
            nn.Linear(32 * 2, 2),
        )

    def forward(self, input):
        encoded_input = self.tokenize(input)
        output = self.pretrained_model(**encoded_input, return_dict=True)
        output = output.last_hidden_state

        output = output[:, 0, :]

        output = output.unsqueeze(0)

        # print(colored('lstm input', 'red'), colored(output.size(), 'green'))

        output, hidden = self.lstm(output)

        # print(colored('lstm output', 'red'), colored(output.size(), 'green'))

        output = output[:, -1, :]

        # print(colored('Last lstm output', 'red'), colored(output.size(), 'green'))

        output = self.classifier(output)

        # print(colored('output', 'red'), colored(output.size(), 'green'))

        return F.log_softmax(output)
