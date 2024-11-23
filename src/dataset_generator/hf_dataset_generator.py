from config import SYNTHETIC_DATASET_GENERATOR_ATTRIBUTES
from src.dataset_generator import DatasetGenerator
from langchain_huggingface import HuggingFaceEndpoint
from collections import defaultdict
from tqdm import tqdm
from random import choice, sample

PROMPT_TEMPLATE="""
<|system|>
Generate a clear, unique, and realistic text for the specified class, keeping the supplied context in mind. Do not explicitly mention the class name, any other classes, or unrelated details. 

### Constraints:
- Length: Strictly **limit the response to a maximum of {number_of_words} words.** Shorter, concise texts are encouraged.
- Style: Write confidently and in a {attribute} manner.
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
1. Ensure the text is concise and strictly within {number_of_words} words.
2. Focus solely on the target class, ensuring clarity and alignment with the context.
3. Use your assigned style ({attribute}) while maintaining coherence.
4. Avoid introducing ambiguity that could link the text to other classes.
5. Output only the generated text without additional indicators or comments.

</s>
<|assistant|>
"""


class HFDatasetGenerator(DatasetGenerator):
    def __init__(self, prompt_template:str, model_repo:str) -> None:
        super().__init__(prompt_template)
        self.llm = HuggingFaceEndpoint(repo_id=model_repo, max_new_tokens=1000, repetition_penalty=1.03, timeout = 300)
         
    def generate(self, context:str, classes:list[str], samples_per_class:int = 5, number_of_words:int = 50) -> dict[str,str]:
        responses = defaultdict(list)
            
        for specific_class in classes:
            for _ in tqdm(range(samples_per_class), desc=f"Generating synthetic data for class {specific_class}"):
                attribute = choice(SYNTHETIC_DATASET_GENERATOR_ATTRIBUTES)

                examples = sample(responses[specific_class], min(5,len(responses[specific_class])))

                prompt = self.prompt_template.format(
                    attribute=attribute,
                    context=context,
                    classes=classes,
                    specific_class=specific_class,
                    number_of_words=number_of_words,
                    examples=examples)
                
                response = self.llm.invoke(prompt)

                responses[specific_class].append(response.strip())
        
        return responses