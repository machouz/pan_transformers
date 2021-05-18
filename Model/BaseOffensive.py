from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
import Input
import Output
import torch

# Load model
model = AutoModelForSequenceClassification.from_pretrained(
    "cardiffnlp/twitter-roberta-base-offensive"
)
tokenizer = AutoTokenizer.from_pretrained(
    "cardiffnlp/twitter-roberta-base-offensive")

# Fine tunning

# Save model
model.save_pretrained("cardiffnlp/twitter-roberta-base-offensive")
tokenizer.save_pretrained("cardiffnlp/twitter-roberta-base-offensive")
