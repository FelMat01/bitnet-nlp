from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    TextClassificationPipeline
)
from datasets import Dataset
from pathlib import Path
from src.classifier import Classifier
import pandas as pd
       

class BertClassifier(Classifier):
    def __init__(
        self, labels: list[str], model_name: str = "bert-base-uncased"
    ) -> None:
        self.labels = labels
        id2label = {i: label for i, label in enumerate(labels)}
        label2id = {label: i for i, label in enumerate(labels)}
        config = AutoConfig.from_pretrained(
            model_name, num_labels=len(labels), id2label=id2label, label2id=label2id
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
            "num_train_epochs": 3,
            "learning_rate": 5e-3,
            # "fp16": True,  # Use mixed precision training to save memory
            "logging_dir": "./logs",
            "logging_steps": 100,
            "save_steps": 500,
            "save_total_limit": 2,  # Keep only the last
        }

    def predict(self, str: str) -> str:
        pipeline = TextClassificationPipeline(model=self.model, tokenizer=self.tokenizer)
        return pipeline(str)[0]["label"]

    def train(self, data: dict[str, list[str]], **training_config):
        self._load_dataset(data)

        config = self.default_training_config.copy()
        config.update(training_config)

        # Define training arguments
        training_args = TrainingArguments(**config)

        # Initialize Trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_dataset,
            tokenizer=self.tokenizer,
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
        self.tokenized_input = self.tokenizer(
            self.dataset["text"], truncation=True, padding=True
        )
        self.tokenized_dataset = self.dataset.map(
            self._preprocess_dataset, batched=True
        )

    def _preprocess_dataset(self, dataset):
        return self.tokenizer(dataset["text"], padding="max_length", truncation=True)
