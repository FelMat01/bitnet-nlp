from pathlib import Path
from datasets import Dataset
import pandas as pd
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from tqdm import tqdm

class Classifier:
    def __init__(self, labels: list[str], model_name: str) -> None:
        pass
    def predict(self, str: str) -> str:
        pass
    def train(self, data: dict[str, list[str]], dataset: Dataset) -> None:
        pass
    def export(self, dir: Path) -> None:
        pass
    def load(self, load_path: Path) -> None:
        pass

    def build_dataset(self, data: dict[str, list[str]]):
        df = pd.DataFrame(data)
        df = df.melt(var_name="labels", value_name="text")
        df = df[df["labels"].isin(self.labels)]
        dataset = Dataset.from_pandas(df)
        dataset = dataset.class_encode_column('labels')

        split_ds = dataset.train_test_split(test_size=0.2, stratify_by_column="labels")
        train = split_ds['train']
        test = split_ds['test']

        split_ds = train.train_test_split(test_size=0.2, stratify_by_column="labels")
        self.train_dataset = split_ds['train']
        self.val_dataset = split_ds['test']
        self.test_dataset = test
        return split_ds
    

    def evaluate_classifier(self, df, label_key='class', show_progress=False, convert_to_numbers=False):
        if show_progress:
            predicted = []
            for text in tqdm(df['text'], desc="Predicting"):
                predicted.append(self.predict(text))
            df['predicted_class'] = predicted
        else:
            df['predicted_class'] = df['text'].apply(self.predict)

        if convert_to_numbers:
            y_true = df[label_key].apply(lambda x: self.labels.index(x))
            y_pred = df['predicted_class'].apply(lambda x: self.labels.index(x))
        else:
            y_true = df[label_key]
            y_pred = df['predicted_class']

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        return accuracy, precision, recall, f1
