import joblib
from pathlib import Path

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src.classifier import Classifier

class NaiveBayesClassifier(Classifier):
    def __init__(self, labels: list[str]) -> None:
        self.labels = labels
        self.pipeline = Pipeline([
            ('vectorizer', CountVectorizer()),  # Converts text to a bag-of-words representation
            ('classifier', MultinomialNB())    # Naive Bayes classifier
        ])

    def train(self, data: dict[str, list[str]]) -> None:
        """
        Trains the classifier with labeled data.
        Args:
            data: A dictionary where keys are label names and values are lists of text examples for that label.
        """
        texts = []
        labels = []
        for label, examples in data.items():
            if label not in self.labels:
                raise ValueError(f"Unknown label: {label}")
            texts.extend(examples)
            labels.extend([label] * len(examples))
        
        self.pipeline.fit(texts, labels)

    def predict(self, text: str) -> str:
        """
        Predicts the label for a given text.
        Args:
            text: The text string to classify.
        Returns:
            The predicted label.
        """
        return self.pipeline.predict([text])[0]

    def export(self, dir: Path) -> None:
        """
        Exports the trained model to a file.
        Args:
            dir: The directory where the model will be saved.
        """
        dir.mkdir(parents=True, exist_ok=True)
        model_path = dir / "naive_bayes_model.pkl"
        joblib.dump(self.pipeline, model_path)

    def load(self, load_path: Path) -> None:
        """
        Loads a trained model from a file.
        Args:
            load_path: The path to the model file.
        """
        self.pipeline = joblib.load(load_path)
