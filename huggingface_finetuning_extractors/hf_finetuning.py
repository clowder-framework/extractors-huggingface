import numpy as np
import evaluate
from datasets import load_dataset
import torch
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer
import sys

import ray.train.huggingface.transformers
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig, RunConfig
from ray.air.integrations.wandb import WandbLoggerCallback


def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    metric = evaluate.load("accuracy")
    return metric.compute(predictions=predictions, references=labels)


class HuggingFaceSequenceClassificationFineTuner:
    def __init__(self, model_name, num_labels, hf_dataset, num_train_examples, num_test_examples,
                 model_file_name="model", num_workers=1, use_gpu=False, wandb_project=None):

        self.model_name = model_name
        self.num_labels = num_labels
        self.model_file_name = model_file_name

        # Dataset could be either from hugging face or local
        # If using huggingface dataset
        self.hf_dataset = hf_dataset
        self.num_train_examples = num_train_examples
        self.num_test_examples = num_test_examples

        # If using wandb
        self.wandb_project = wandb_project

        # If using GPU
        self.use_gpu = use_gpu
        # Set number of workers
        self.num_workers = num_workers

        self.train_ref = None
        self.test_ref = None

    def load_dataset(self):
        dataset = load_dataset(self.hf_dataset).shuffle()
        if self.num_train_examples:
            train_dataset = dataset["train"].select(range(self.num_train_examples))
        else:
            train_dataset = dataset["train"]
        if self.num_test_examples:
            test_dataset = dataset["test"].select(range(self.num_test_examples))
        else:
            test_dataset = dataset["test"]

        # Put datasets in Ray's object store
        self.train_ref = ray.put(train_dataset)
        self.test_ref = ray.put(test_dataset)

    def train_func(self):
        # Load the dataset
        train_dataset = ray.get(self.train_ref)
        test_dataset = ray.get(self.test_ref)

        # Load the model
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=self.num_labels)
        # Load the tokenizer and tokenize the data
        # Tokenize and prepare datasets
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        train_dataset = train_dataset.map(lambda examples: tokenize_function(examples, tokenizer), batched=True)
        test_dataset = test_dataset.map(lambda examples: tokenize_function(examples, tokenizer), batched=True)

        # Define the training arguments
        training_args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            report_to="none"
        )

        # Define the trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics
        )

        # Adjusting for Ray
        callback = ray.train.huggingface.transformers.RayTrainReportCallback()
        trainer.add_callback(callback)
        trainer = ray.train.huggingface.transformers.prepare_trainer(trainer)

        # Train the model
        trainer.train()

    def run(self):

        self.load_dataset()

        scaling_config = ScalingConfig(
            num_workers=self.num_workers,
            use_gpu=self.use_gpu
        )

        callbacks = []

        if self.wandb_project:
            callbacks.append(WandbLoggerCallback(project=self.wandb_project))

        ray_trainer = TorchTrainer(
            self.train_func,
            scaling_config=scaling_config,
            run_config=RunConfig(
                callbacks=callbacks
            )
        )

        result = ray_trainer.fit()
        # Save model
        # TODO: Metric for choosing the best checkpoint could be a parameter
        best_ckpt = result.get_best_checkpoint(metric="eval_loss", mode="min")
        with best_ckpt.as_directory() as checkpoint_dir:
            model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir+"/checkpoint")
            filename = self.model_file_name + ".pt"
            torch.save(model, filename)

        # Stop ray
        ray.shutdown()

        return result


# Testing the class
if __name__ == "__main__":
    model_name = "distilbert-base-cased"
    hf_dataset = "yelp_review_full"
    num_labels = 5
    num_train_examples = 3
    num_test_examples = 3
    use_gpu = False
    num_workers = 1

    finetuner = HuggingFaceSequenceClassificationFineTuner(model_name=model_name, num_labels=num_labels,
                                                           hf_dataset=hf_dataset,
                                                           num_train_examples=num_train_examples,
                                                           num_test_examples=num_test_examples,
                                                           use_gpu=use_gpu,
                                                           num_workers=num_workers)
    finetuner.run()
