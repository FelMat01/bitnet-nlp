from src.dataset_generator import DatasetGenerator
from collections import defaultdict
from tqdm import tqdm
from openai import OpenAI
import json

PROMPT_TEMPLATE =""" **Instructions:**
- **Do not** mention the class name in the paragraph.
- Base the paragraph on the following context.
- **Do not** repeat or use similar content from the provided examples.

**Context:**
{context}

**Class:**
{specific_class}

**Examples:**
{examples}

If not examples were provided, you have to generate the seed examples, so try to be rich in differences and use of important words.

**Tasks:**
1. **Elements to Exclude:** Create a list of elements from the context that should **not** be repeated in the paragraph.
2. **Alternative Expressions:** Identify alternative expressions or synonyms that can replace the class name.
3. **Related Words:** Identify words related with the class in the context, like objects, details.
3. **Generated Paragraphs:** Provide a list of at least 10 example paragraphs following the same guidelines. In python list[str] format.

**Output Format:**
1. **Elements to Exclude:**
   - Item 1
   - Item 2
   - ...

2. **Alternative Expressions for the Class:**
   - Expression 1
   - Expression 2
   - ...

3. **Generated Paragraph:**
   ["paragraph1", "paragraph"]
   
"""
def parse_to_list(input_string):
    """
    Parse the given string into a Python list of strings, removing newline characters.
    
    Args:
        input_string (str): The input string containing text to be converted into a list.

    Returns:
        list: A list of strings parsed from the input string, with newlines removed.
    """
    try:
        # Extract the JSON array part from the input
        json_part = input_string.split('[', 1)[-1].rsplit(']', 1)[0]
        # Replace newlines and extra spaces
        json_cleaned = json_part.replace('\n', '').strip()
        # Load the cleaned JSON content
        return json.loads(f"[{json_cleaned}]")
    except json.JSONDecodeError as e:
        print("Error parsing the input string:", e)
        return []
    
class OpenAIDatasetGenerator(DatasetGenerator):
    def __init__(self, prompt_template:str) -> None:
        super().__init__(prompt_template)
        self.client = OpenAI()
         
    def generate(self, context:str, classes:list[str], samples_per_class:int = 5, number_of_words:int = 50) -> dict[str,str]:
        responses = defaultdict(list)
            
        for specific_class in classes:
            examples= ""
            for _ in tqdm(range(samples_per_class), desc=f"Generating synthetic data for class {specific_class}"):
                prompt = self.prompt_template.format(context=context, specific_class=specific_class, number_of_words=number_of_words, examples=examples)
                message = [{
                                "role": "system", 
                                "content": [
                                    {
                                    "type": "text",
                                    "text": " You are an expert in creating examples for text classification datasets. Your task is to generate highly diverse and unique examples that cover various patterns and nuances to improve the dataset's quality. Ensure that the examples are distinct from each other and encompass a wide range of scenarios relevant to the classification task."
                                    }],
                                },
                                {
                                "role": "user",
                                "content": [
                                    {
                                    "type": "text",
                                    "text": prompt
                                    }
                                ]
                                }
                            ]
                completion = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=message
                )

                response = completion.choices[0].message.content
                list_response = parse_to_list(response)
                responses[specific_class].extend(list_response)
                for resp in list_response:
                    
                    examples += f"{resp}\n\n"
        
        return responses
    
    

