from pathlib import Path
from src.classifier import Classifier
from scipy.spatial.distance import cdist
import numpy as np
from collections import Counter
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score


class KNNClassifier():
    def __init__(self, hf=None):
        self.embeddings = None
        self.hf = hf
    def set_hf_endpoint(self):
        from langchain_huggingface import HuggingFaceEndpointEmbeddings
        model = "sentence-transformers/all-MiniLM-L6-v2"
        self.hf = HuggingFaceEndpointEmbeddings(
            model=model,
            task="feature-extraction",
            huggingfacehub_api_token="",
        )

    def train(self, data: dict[str, list[str]]):

            self.embeddings = []
            self.labels = []
            for class_name, text in data.items():
                self.embeddings.append(self.hf.embed_documents(text))
                self.labels.extend([class_name] * len(text))
            self.embeddings = np.vstack(self.embeddings)

            return
       
    def predict(self, text):
        new_sentence_embedding = self.hf.embed_documents([text])[0]
        distances = cdist([new_sentence_embedding], self.embeddings, metric='cosine')[0]


        nearest_indices = np.argsort(distances)[:5]
        nearest_labels = [self.labels[idx] for idx in nearest_indices]


        most_common_label = Counter(nearest_labels).most_common(1)[0][0]

        return most_common_label

    from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

    def evaluate_classifier(self, df):
        df['predicted_class'] = df['text'].apply(self.predict)

        y_true = df['class']
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



