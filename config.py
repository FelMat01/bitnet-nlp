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
SYNTHETIC_DATASET_GENERATOR_PROMPT="""
<|system|>
Generate a clear, unique, and realistic paragraph for the specified class. Do not explicitly mention the class name, any other classes, or unrelated details.
The text must distinctly align with the target class without any ambiguity or overlap with other classes.

### Constraints:
- Length: Maximum {number_of_words} words.
- Style: Write confidently and in a {attribute} manner, as if you are knowledgeable in the subject.
- Originality: Avoid rephrasing or mimicking examples provided. Ensure the tone and content differ from the examples.

### Context:
{context}

### Available Classes:
{classes}

### Target Class:
{specific_class}

### Examples:
{examples}

### Instructions:
1. Focus solely on the target class, ensuring clarity and alignment with the context.
2. Use your assigned style ({attribute}) while maintaining professionalism and coherence.
3. Avoid introducing ambiguity that could link the paragraph to other classes.
4. Only output the generated paragraph without additional indicators or comments.

</s>
<|assistant|>
"""

# Models
MODELS_FOLDER = Path("./models")