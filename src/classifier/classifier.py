from pathlib import Path
from datasets import Dataset
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