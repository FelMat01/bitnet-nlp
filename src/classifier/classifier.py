from pathlib import Path

class Classifier:
    def __init__(self, labels: list[str], model_name: str) -> None:
        pass
    def predict(self, str: str) -> str:
        pass
    def train(self) -> None:
        pass
    def export(self, dir: Path) -> None:
        pass
    def load(self, load_path: Path) -> None:
        pass