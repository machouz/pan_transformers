import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import softmax


class DefaultModel(pl.LightningModule):
    def tokenize(self, input):
        return self.tokenizer(input, padding='max_length', truncation=True, return_tensors="pt").to(self.device)

    def common_step(self, batch, batch_idx):
        batch = batch[0]

        y = batch["label"]
        x = batch["input"]

        output = self.forward(x)
        loss = F.cross_entropy(output, y.view(-1))

        return output, y, loss

    def training_step(self, batch, batch_idx):
        output, y, loss = self.common_step(batch, batch_idx)

        self.log('train_loss', loss, on_step=False, on_epoch=True, logger=True)

        y_hat = output.argmax()
        corrects = (y_hat == y)
        accuracy = corrects.sum().float()

        self.log('train_accuracy', accuracy,
                 on_step=False, on_epoch=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        output, y, loss = self.common_step(batch, batch_idx)

        self.log('val_loss', loss, on_step=False, on_epoch=True, logger=True)

        y_hat = output.argmax()
        corrects = (y_hat == y)
        accuracy = corrects.sum().float()

        self.log('val_accuracy', accuracy,
                 on_step=False,  on_epoch=True, logger=True)

        return loss

    def configure_optimizers(self):
        param_optimizer = list(self.named_parameters())
        no_decay = ["bias", "gamma", "beta"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay_rate": 0.01
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay_rate": 0.0
            },
        ]
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=2e-5,
        )
        return optimizer
