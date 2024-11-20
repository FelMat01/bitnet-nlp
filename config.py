from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Synthetic Data
SYNTHETIC_DATASET_GENERATE = True
SYNTHETIC_DATASET_FOLDER = Path("./data")
SYNTHETIC_DATASET_PATH = Path(f"{SYNTHETIC_DATASET_FOLDER}/synthetic_dataset.json")
SYNTHETIC_DATASET_GENERATOR_ATTRIBUTES = [
    "happy",
    "professional",
    "interested",
    "scientist",
]
SYNTHETIC_DATASET_GENERATOR_PROMPT="""
<|system|>
Generate a paragraph for the class WITHOUT NAMING THE CLASS or other classes based on the context.
The generated text must not be ambiguous with toher classes.

Write a maximum of {number_of_words} words.
Write it as if you where knowledgable and {attribute}.
Try to be different from examples (if present).

Context:
{context}

All Classes:
{classes}

Current Class: 
{specific_class}

Examples (personalities in examples are not equal to your own personality):
{examples}

Note: Only respond with the requested text without any normal conversation indicators.

</s>
<|assistant|>
"""

# Models
MODELS_FOLDER = Path("./models")