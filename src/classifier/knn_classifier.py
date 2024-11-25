from src.classifier import Classifier
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from typing import Union

class KNNClassifier(Classifier):
    def __init__(self, labels: list[str], hf=None):
        self.embeddings = None
        self.hf = hf
        labels.sort()
        self.labels = labels
    
    def set_hf_endpoint(self):
        from langchain_huggingface import HuggingFaceEndpointEmbeddings
        model = "sentence-transformers/all-MiniLM-L6-v2"
        self.hf = HuggingFaceEndpointEmbeddings(
            model=model,
            task="feature-extraction",
            huggingfacehub_api_token="",
        )

    def train(self, data: dict[str, list[str]], dataset=None):
        if self.hf is None:
            self.set_hf_endpoint()

        self.embeddings = []
        self.embeddings_labels = []
        for class_name, text in tqdm(data.items(), desc="Embedding documents"):
            self.embeddings.append(self.hf.embed_documents(text))
            self.embeddings_labels.extend([class_name] * len(text))
        self.embeddings = np.vstack(self.embeddings)

        self.knn = KNeighborsClassifier(n_neighbors=5, metric='cosine')
        self.knn.fit(self.embeddings, self.embeddings_labels)
       
    def predict(self, texts: Union[str, list[str]], return_index: bool = False):
        if self.hf is None:
            self.set_hf_endpoint()

        if isinstance(texts, str):
            texts = [texts]
        embeddings = self.hf.embed_documents(texts)
        predictions = self.knn.predict(embeddings)
        prediction = str(predictions[0])
        if return_index:
            return self.labels.index(prediction)
        return prediction
    def export(self, dir):
        print("KNN is non-exportable since it has no weights")
