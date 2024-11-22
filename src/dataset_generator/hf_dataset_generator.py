from config import SYNTHETIC_DATASET_GENERATOR_ATTRIBUTES
from src.dataset_generator import DatasetGenerator
from langchain_huggingface import HuggingFaceEndpoint
from collections import defaultdict
from tqdm import tqdm
from random import choice, sample

PROMPT_TEMPLATE="""
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


class HFDatasetGenerator(DatasetGenerator):
    def __init__(self, model_repo:str) -> None:
        self.llm = HuggingFaceEndpoint(repo_id=model_repo, max_new_tokens=1000, repetition_penalty=1.03, timeout = 300)
         
    def generate(self, context:str, classes:list[str], samples_per_class:int = 5, number_of_words:int = 50) -> dict[str,str]:
        responses = defaultdict(list)
            
        for specific_class in classes:
            for _ in tqdm(range(samples_per_class), desc=f"Generating synthetic data for class {specific_class}"):
                attribute = choice(SYNTHETIC_DATASET_GENERATOR_ATTRIBUTES)

                examples = sample(responses[specific_class], min(5,len(responses[specific_class])))

                prompt = PROMPT_TEMPLATE.format(
                    attribute=attribute,
                    context=context,
                    classes=classes,
                    specific_class=specific_class,
                    number_of_words=number_of_words,
                    examples=examples)
                
                response = self.llm.invoke(prompt)

                responses[specific_class].append(response.strip())
        
        return responses