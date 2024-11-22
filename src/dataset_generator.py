from langchain_huggingface import HuggingFaceEndpoint
from collections import defaultdict
from tqdm import tqdm
from config import SYNTHETIC_DATASET_GENERATOR_ATTRIBUTES, SYNTHETIC_DATASET_GENERATOR_PROMPT
from random import choice, sample

class DatasetGenerator:
    def __init__(self, model_repo:str) -> None:
        self.llm = HuggingFaceEndpoint(repo_id=model_repo, max_new_tokens=1000, repetition_penalty=1.03, timeout = 300)
         
    def generate(self, context:str, classes:list[str], samples_per_class:int = 5, number_of_words:int = 50) -> dict[str,str]:
        responses = defaultdict(list)
            
        for specific_class in classes:
            for _ in tqdm(range(samples_per_class), desc=f"Generating synthetic data for class {specific_class}"):
                attribute = choice(SYNTHETIC_DATASET_GENERATOR_ATTRIBUTES)

                examples = sample(responses[specific_class], min(5,len(responses[specific_class])))

                prompt = SYNTHETIC_DATASET_GENERATOR_PROMPT.format(
                    attribute=attribute,
                    context=context,
                    classes=classes,
                    specific_class=specific_class,
                    number_of_words=number_of_words,
                    examples=examples)
                
                response = self.llm.invoke(prompt)

                responses[specific_class].append(response.strip())
        
        return responses