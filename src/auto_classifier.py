from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score
import numpy as np


def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {"accuracy": accuracy_score(labels, predictions)}

class AutoClassifier:
    def __init__(self, model_name, num_labels):
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=self.num_labels)
    
    def encode_dataset(self, dataset):
        return dataset.map(self.preprocess_dataset, batched=True)

    def fine_tune(self, encoded_dataset, epochs=3):
        # Rename label column to 'labels' as expected by Trainer
        encoded_dataset = encoded_dataset.rename_column("label", "labels")

        # Split dataset
        self.train_dataset = encoded_dataset["train"].shuffle(seed=42).select(range(20000))
        self.val_dataset = encoded_dataset["test"].shuffle(seed=42).select(range(5000))

        # Define training arguments
        training_args = TrainingArguments(
            auto_find_batch_size=True,
            output_dir="./results",
            overwrite_output_dir=True,
            #per_device_train_batch_size=1,  # Reduce if CUDA memory error persists
            per_device_eval_batch_size=1,
            #gradient_accumulation_steps=4,  # Accumulate gradients to simulate larger batch size
            num_train_epochs=epochs,
            learning_rate=5e-3,
            fp16=True,  # Use mixed precision training to save memory
            logging_dir="./logs",
            logging_steps=100,
            save_steps=500,
            save_total_limit=2,  # Keep only the last 2 model checkpoints
        )

        # Initialize Trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics
        )

        # Train the model
        self.trainer.train()
        self.trainer.evaluate()

        # Save the model
        self.model.save_pretrained("./bert-sequence-classification")
        self.tokenizer.save_pretrained("./bert-sequence-classification")

    def preprocess_dataset(self, dataset):
        return self.tokenizer(dataset['text'], padding="max_length", truncation=True)
