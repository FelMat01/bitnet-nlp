from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
from datasets import Dataset
from src.download_datasets import get_news_group_dataset
import pandas as pd

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

df = get_news_group_dataset()

# Convert classes to integer labels
df['class_int'] = pd.factorize(df['class'])[0]

train_dataset = Dataset.from_dict({
    "sentence": df["text"].tolist(),
    "label": df["class_int"].tolist(),
})

loss = losses.BatchAllTripletLoss(model)

trainer = SentenceTransformerTrainer(
    model=model,
    train_dataset=train_dataset,
    loss=loss,
    output_path="models/triplet_model"
)
trainer.train()


# Group by 'class' and print one example from each class
for _, group in df.groupby('class'):
    class_name = group.iloc[0]['class']
    class_int = group.iloc[0]['class_int']
    example_text = group.iloc[0]['text']
    print(f"Class: {class_name} (Int: {class_int})\nExample: {example_text}\n")


