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
parser.add_argument("-mp", "--model_path",
                    help="Model CKPT path", required=False)

parser.add_argument("-i", "--input_path",
                    help="Data to predict", required=False)
parser.add_argument("-o", "--output_path",
                    help="Path for the prediction", required=False)


def get_model_prediction(model, input):
    output = model(input)
    y_hat = output.argmax()
    return int(y_hat)


if __name__ == "__main__":
    args = parser.parse_args()

    if args.input_path:
        input_path = args.input_path
    else:
        input_path = f"data_test/{args.language}/"

    try:
        pretrained_model = args.pretrained_mode if args.pretrained_mod else PRETRAINED_MODEL[
            args.language]
    except AttributeError:
        pretrained_model = PRETRAINED_MODEL[args.language]

    if args.model_path:
        model_path = args.model_path
    else:
        models_path = f'lightning_logs/{args.language}/{pretrained_model.split("/")[-1]}/{args.model}'
        model_path = glob.glob(models_path + '/**/*.ckpt', recursive=True)[0]

    if args.output_path:
        output_path = args.output_path
    else:
        output_path = os.path.dirname(model_path)

    model = MODEL_MAP[args.model].load_from_checkpoint(
        checkpoint_path=model_path, pretrained_model_name=pretrained_model)
    data = Input.get_data_dict(input_path, with_label=False)

    Path(output_path).mkdir(parents=True, exist_ok=True)

    for feed in tqdm(data):
        outputResult(
            id=feed['id'],
            type=get_model_prediction(model, feed['input']),
            lang=args.language,
            prepath=output_path
        )
