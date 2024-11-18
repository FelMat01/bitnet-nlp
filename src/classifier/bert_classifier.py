from pathlib import Path
from src.classifier import Classifier

class BertClassifier(Classifier):
    def __init__(self, labels : list[str]):
        pass
    def predict(self, str) -> str:
        pass
    def train(self):
        pass
    def export(self, output_dir : Path):
        pass
    def load(self, load_path : Path):
        pass