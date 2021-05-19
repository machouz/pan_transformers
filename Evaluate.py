import argparse
import glob
import os
from pathlib import Path

from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm

import Input
from Model.ConvolutionalHate import ConvolutionalHate
from Model.CustomHate import CustomHate
from Model.LSTMHate import LSTMHate
from Output import outputResult

PRETRAINED_MODEL = {
    'en': 'bert-base-cased',
    'es': 'dccuchile/bert-base-spanish-wwm-cased'
}

MODEL_MAP = {
    'CustomHate': CustomHate,
    'LSTMHate': LSTMHate,
    'ConvolutionalHate': ConvolutionalHate,
}


parser = argparse.ArgumentParser(description="Feed classifier")
parser.add_argument("-l", "--language", help="Language", required=True)
parser.add_argument("-p", "--pretrained_model",
                    help="Pretrained Model", required=False)
parser.add_argument("-m", "--model", help="Model", required=True)


def get_model_prediction(model, input):
    output = model(input)
    y_hat = output.argmax()
    return int(y_hat)


if __name__ == "__main__":
    args = parser.parse_args()
    data_path = f"data_test/{args.language}/"
    try:
        pretrained_model = args.pretrained_mode
    except AttributeError:
        pretrained_model = PRETRAINED_MODEL[args.language]
    models_path = f'lightning_logs/{args.language}/{args.model}'
    files = glob.glob(
        models_path + '/**/*.ckpt', recursive=True)
    for file in files:
        model = MODEL_MAP[args.model].load_from_checkpoint(
            checkpoint_path=file, pretrained_model_name=pretrained_model)
        data = Input.get_data_dict(data_path, with_label=False)
        folder = f"{os.path.dirname(file) }/prediction/"
        Path(folder).mkdir(parents=True, exist_ok=True)
        for feed in tqdm(data):
            outputResult(
                id=feed['id'],
                type=get_model_prediction(model, feed['input']),
                lang=args.language,
                prepath=folder
            )
