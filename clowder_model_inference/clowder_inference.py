import numpy as np

import torch
from transformers import AutoTokenizer, pipeline
import pandas as pd
import sys


def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], padding="max_length", truncation=True, return_tensors="pt")


# Assuming dataset is in a csv file
class TorchSQModelPredictor:
    def __init__(self, foundation_model_name, model_path, dataset_path, prediction_file_name):
        self.model = torch.load(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(foundation_model_name)
        self.pipe = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer)
        self.dataset_path = dataset_path
        self.prediction_file_name = prediction_file_name

    def predict(self):
        # Load dataset
        dataset = pd.read_csv(self.dataset_path)
        predictions = self.pipe(dataset["text"].tolist(), truncation=True, padding="max_length")

        # Add prediction and confidence to the dataset
        dataset["predictions"], dataset["prediction_confidence"] = zip(*[(p['label'], p['score']) for p in predictions])

        dataset.to_csv(self.prediction_file_name + ".csv", index=False)

        return True


if __name__ == "__main__":
    foundation_model_name = "distilbert-base-cased"
    model_path = "model.pt"
    dataset_path = "data/test.csv"
    predictor = TorchSQModelPredictor(foundation_model_name, model_path, dataset_path)
    predictions = predictor.predict()
