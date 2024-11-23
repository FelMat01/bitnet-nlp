from pathlib import Path
from datasets import Dataset
import pandas as pd
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
