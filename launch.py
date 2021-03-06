from FeedData import FeedData
from Model.CustomHate import CustomHate
from Model.CustomMaxHate import CustomMaxHate
from Model.LSTMHate import LSTMHate
from Model.ConvolutionalHate import ConvolutionalHate
from scipy.special import softmax
import pandas as pd
import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
import argparse


LANGUAGE = 'en'
PRETRAINED_MODEL = {
    'en': 'bert-base-cased',
    'es': 'dccuchile/bert-base-spanish-wwm-cased'
}

MODEL_MAP = {
    'CustomHate': CustomHate,
    'CustomMaxHate': CustomMaxHate,
    'LSTMHate': LSTMHate,
    'ConvolutionalHate': ConvolutionalHate,
}


parser = argparse.ArgumentParser(description="Feed classifier")
parser.add_argument("-l", "--language", help="Language", required=True)
parser.add_argument("-p", "--pretrained_model",
                    help="Pretrained Model", required=False)
parser.add_argument("-m", "--model", help="Model", required=True)
parser.add_argument("-mp", "--model_path", help="Model path", required=False)


if __name__ == "__main__":
    args = parser.parse_args()
    path = f"data/{args.language}/"

    try:
        pretrained_model = args.pretrained_model if args.pretrained_model else PRETRAINED_MODEL[
            args.language]
    except AttributeError:
        pretrained_model = PRETRAINED_MODEL[args.language]

    if args.model_path:
        model = MODEL_MAP[args.model].load_from_checkpoint(
            checkpoint_path=args.model_path, pretrained_model_name=pretrained_model)
    else:
        model = MODEL_MAP[args.model](pretrained_model)

    data = FeedData(path)

    tb_logger = pl_loggers.TensorBoardLogger(
        f"lightning_logs/{args.language}/{pretrained_model.split('/')[-1]}", name=args.model)
    trainer = Trainer(gpus=[1], auto_select_gpus=True,
                      logger=tb_logger, max_epochs=10000)
    trainer.fit(model, data)
