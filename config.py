from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Synthetic Data
SYNTHETIC_DATASET_GENERATE = False
SYNTHETIC_DATASET_FOLDER = Path("./data")
SYNTHETIC_DATASET_PATH = Path(f"{SYNTHETIC_DATASET_FOLDER}/synthetic_dataset.json")
SYNTHETIC_DATASET_GENERATOR_ATTRIBUTES = [
    "happy",
    "professional",
    "interested",
    "scientist",
]

# Models
MODELS_FOLDER = Path("./models")