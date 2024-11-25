from src.classifier import Classifier
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from typing import Union
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.lines import Line2D

class KNNClassifier(Classifier):
    def __init__(self, labels: list[str], hf=None):
        self.embeddings = []
        self.embeddings_labels = []
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


def compute_pca_and_colors(labels, embeddings):
    """
    Computes PCA and assigns colors based on unique labels.
    
    Parameters:
    - labels: List of class labels (strings).
    - embeddings: List of embeddings (2D array-like).
    
    Returns:
    - pca_result: 2D array with PCA-transformed embeddings.
    - c: List of colors corresponding to the labels.
    - label_to_color: Dictionary mapping labels to colors.
    """
    # Generate unique colors for the number of unique labels
    unique_labels = list(set(labels))
    colors = plt.cm.tab10.colors[:len(unique_labels)]  # Use tab10 colormap for up to 10 colors
    label_to_color = {label: colors[i % len(colors)] for i, label in enumerate(unique_labels)}
    
    # Map each label to a color
    c = [label_to_color[label] for label in labels]
    
    # Perform PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(embeddings)
    
    return pca_result, c, label_to_color

def plot_pca_with_streamlit(labels, embeddings, title="PCA Visualization"):
    """
    Plots PCA-transformed embeddings with colors corresponding to the provided labels.
    
    Parameters:
    - labels: List of class labels (strings).
    - embeddings: List of embeddings (2D array-like).
    - title: Title of the plot (default: "PCA Visualization").
    
    Displays the plot in Streamlit.
    """
    # Compute PCA and colors (cached to avoid recomputation)
    pca_result, c, label_to_color = compute_pca_and_colors(labels, embeddings)
    
    # Plot
    plt.figure(figsize=(10, 7))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=c, cmap='tab10')
    plt.title(title)
    
    # Create legend
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=label,
                            markerfacecolor=color, markersize=10) 
                    for label, color in label_to_color.items()]
    plt.legend(handles=legend_elements, loc='best')
    
    # Display the plot in Streamlit
    st.pyplot(plt)
    plt.clf()
