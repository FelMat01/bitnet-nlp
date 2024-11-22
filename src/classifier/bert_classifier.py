from pathlib import Path

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    TextClassificationPipeline
)
from datasets import Dataset
from src.classifier import Classifier

import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision = precision_score(labels, predictions, average="macro")
    recall = recall_score(labels, predictions, average="macro")
    accuracy = accuracy_score(labels, predictions)
    return {"precision": precision, "recall": recall, "accuracy": accuracy}

class BertClassifier(Classifier):
    def __init__(
        self, labels: list[str], model_name: str = "bert-base-uncased"
    ) -> None:
        self.labels = labels
        id2label = {i: label for i, label in enumerate(labels)}
        label2id = {label: i for i, label in enumerate(labels)}
        config = AutoConfig.from_pretrained(
            model_name, num_labels=len(labels), id2label=id2label, label2id=label2id, hidden_dropout_prob=0
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, config=config
        )

        self.default_training_config = {
            # Same as below
            "auto_find_batch_size": True,
            "output_dir": "./results",
            "overwrite_output_dir": True,
            # "per_device_train_batch_size": 1,  # Reduce if CUDA memory error persists
            "per_device_eval_batch_size": 1,
            # "gradient_accumulation_steps": 4,  # Accumulate gradients to simulate larger batch size
            "num_train_epochs": 5,
            "learning_rate": 5e-5,
            "lr_scheduler_type": "cosine",
            # "fp16": True,  # Use mixed precision training to save memory
            "logging_dir": "./logs",
            "logging_steps": 10,
            "save_steps": 500,
            "save_total_limit": 2,  # Keep only the last
            "eval_strategy": "epoch"
        }

    def predict(self, str: str) -> str:
        pipeline = TextClassificationPipeline(model=self.model, tokenizer=self.tokenizer)
        return pipeline(str)[0]["label"]

    def train(self, data, dataset: Dataset, **training_config):
        self.train_dataset = dataset['train']
        self.val_dataset = dataset['test']
        self.tokenized_train_dataset = self._tokenize_dataset(self.train_dataset)
        self.tokenized_val_dataset = self._tokenize_dataset(self.val_dataset)

        config = self.default_training_config.copy()
        config.update(training_config)

        # Define training arguments
        training_args = TrainingArguments(**config)

        # Initialize Trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_train_dataset,
            eval_dataset=self.tokenized_val_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics
        )

        # Train the model
        self.trainer.train()

    def export(self, dir: Path) -> None:
        dir.mkdir(parents=True, exist_ok=True)
        model_path = dir / "bert_classifier"
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)

    def load(self, load_path: Path) -> None:
        self.model = AutoModelForSequenceClassification.from_pretrained(load_path)
        self.tokenizer = AutoTokenizer.from_pretrained(load_path)

    def _load_dataset(self, data: dict[str, list[str]]):
        df = pd.DataFrame(data)
        df = df.melt(var_name="labels", value_name="text")
        df = df[df["labels"].isin(self.labels)]
        df["str_labels"] = pd.Categorical(df["labels"], categories=self.labels)
        df["labels"] = df["str_labels"].cat.codes
        self.dataset = Dataset.from_pandas(df)
        self.tokenized_dataset = self.dataset.map(
            self._preprocess_dataset, batched=True
        )

    def _tokenize_dataset(self, dataset: Dataset):
        return dataset.map(self._preprocess_dataset, batched=True)

    def _preprocess_dataset(self, dataset):
        return self.tokenizer(dataset["text"], padding="max_length", truncation=True)
