import kagglehub
import pandas as pd
from pathlib import Path

def get_news_group_dataset():
    kaggle_dataset_folder = kagglehub.dataset_download("jensenbaxter/10dataset-text-document-classification")
    kaggle_dataset_folder_path = Path(kaggle_dataset_folder)
    rows = []
    labels = ["sport", "business", "food", "medical"]
    for label in labels:

        for file_path in (kaggle_dataset_folder_path / label).iterdir():
            if not file_path.is_dir():
                with open(file_path, "r") as file:
                    content = file.read()
                    content = content.strip()
                    # Split the content into lines
                    lines = content.split('\n')

                    # Filter out lines that contain '@'
                    filtered_lines = [line for line in lines if '@' not in line]

                    # Join the filtered lines back together
                    content = '\n'.join(filtered_lines)
                    # Split by words and limit to 50 words
                    words = content.split()
                    if len(words) > 50:
                        content = ' '.join(words[:50])
                    rows.append([content, label])



    df = pd.DataFrame(rows, columns=["text", "class"])
    return df
