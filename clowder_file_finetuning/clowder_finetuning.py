import numpy as np
import evaluate
from datasets import load_dataset
import torch
import os
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer
import sys

import ray.train.huggingface.transformers
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig, RunConfig
from ray.air.integrations.wandb import WandbLoggerCallback
import logging


def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    metric = evaluate.load("accuracy")
    return metric.compute(predictions=predictions, references=labels)


class ClowderSQFineTuner:
    def __init__(self, model_name, num_labels, data_type, local_train_file, local_test_file, ray_storage_path=None,
                 model_file_name="model", num_workers=1, use_gpu=False, wandb_project=None):

        self.model_name = model_name
        self.num_labels = num_labels
        self.model_file_name = model_file_name

        self.data_type = data_type
        self.local_train_file = local_train_file
        self.local_test_file = local_test_file

        # If using wandb
        self.wandb_project = wandb_project

        # If using GPU
        self.use_gpu = use_gpu
        # Set number of workers
        self.num_workers = num_workers

        self.train_ref = None
        self.test_ref = None

        if ray_storage_path:
            self.ray_storage_path = ray_storage_path
        else:
            # Default path
            if not os.path.exists(".clowder_finetuning"):
                self.ray_storage_path = "~/ray_results/"

    def load_dataset(self):
        # Load datasets based on configuration
        dataset = load_dataset(self.data_type,
                               data_files={"train": self.local_train_file, "test": self.local_test_file})
        train_dataset = dataset["train"]
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
            output_dir= self.ray_storage_path + "./results",
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
        try:
            self.load_dataset()

            scaling_config = ScalingConfig(
                num_workers=self.num_workers,
                use_gpu=self.use_gpu
            )

            callbacks = []

            if self.wandb_project:
                callbacks.append(WandbLoggerCallback(project=self.wandb_project))

            # Define the running configuration
            running_config = RunConfig(
                callbacks=callbacks,
                storage_path= self.ray_storage_path + "ray_results"
            )


            ray_trainer = TorchTrainer(
                self.train_func,
                scaling_config=scaling_config,
                run_config=running_config
            )

            result = ray_trainer.fit()
            # Save model
            best_ckpt = result.get_best_checkpoint(metric="eval_loss", mode="min")
            with best_ckpt.as_directory() as checkpoint_dir:
                model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir+"/checkpoint")
                filename = self.model_file_name + ".pt"
                torch.save(model, filename)

            # Stop ray
            ray.shutdown()

            return result
        finally:
            ray.shutdown()


if __name__ == "__main__":
    model_name = "distilbert-base-cased"
    data_type = "csv"
    local_train_file = "data/train_sampled.csv"
    local_test_file = "data/test_sampled.csv"
    ray_storage_path = "/taiga/mohanar2"
    num_labels = 5
    use_gpu = True
    num_workers = 1

    finetuner = ClowderSQFineTuner(model_name=model_name, num_labels=num_labels,
                                   data_type=data_type, ray_storage_path= ray_storage_path,
                                   local_train_file=local_train_file,local_test_file=local_test_file)
    finetuner.run()