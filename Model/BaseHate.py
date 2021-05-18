from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from scipy.special import softmax
import Input
import Output
import torch


# Load model
model = AutoModelForSequenceClassification.from_pretrained(
    "cardiffnlp/twitter-roberta-base-hate"
)
tokenizer = AutoTokenizer.from_pretrained(
    "cardiffnlp/twitter-roberta-base-hate")

# Fine tunning

# Save model
model.save_pretrained("cardiffnlp/twitter-roberta-base-hate")
tokenizer.save_pretrained("cardiffnlp/twitter-roberta-base-hate")
