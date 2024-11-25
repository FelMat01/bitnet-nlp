from src.dataset_generator import DatasetGenerator
from collections import defaultdict
from tqdm import tqdm
from openai import OpenAI
import json
import random
import os
PROMPT_TEMPLATE = """ **Instructions:**
- **Do not** mention the class name in the paragraph.
- Base the paragraph on the following context.
- **Do not** repeat or use similar content from the provided examples.
- Use only the class, but take into account that the classifier for all the classes.
- Who you are influence the data generated.

**Context:**
{context}

**Who you are**
{who_you_are}

**All the classes:**
{classes}

**Class:**
{specific_class}

**Examples:**
{examples}

If not examples were provided, you have to generate the seed examples, so try to be rich in differences and use of important words.

**Tasks:**
1. **Elements to Exclude:** Create a list of elements from the context that should **not** be repeated in the paragraph.
2. **Alternative Expressions:** Identify alternative expressions or synonyms that can replace the class name.
3. **Related Words:** Identify words related with the class in the context, like objects, details.
3. **Generated Paragraphs:** Provide a list of at least {samples_per_inference} example paragraphs following the same guidelines. In python list[str] format.

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

who_you_are_options = [
    "a common guy",
    "a scientist",
    "a curious teenager",
    "an old man",
    "a popular woman",
    "a philosopher",
    "a teacher",
    "a student",
    "a business executive",
    "a journalist",
    "a poet",
    "a skeptic",
    "an artist",
    "a historian",
    "a doctor",
    "a software engineer",
    "a detective",
    "a lawyer",
    "an environmental activist",
    "a sports coach",
    "a famous author",
    "an explorer",
    "a politician",
    "a soldier",
    "a religious leader",
    "a motivational speaker",
    "an adventurer",
    "a comedian",
    "a chef",
    "a farmer",
    "a yoga instructor",
    "a psychologist",
    "a parent",
    "a child",
    "an alien from another planet",
    "a time traveler",
    "a wizard",
    "a stupid person",
    "a person with short memory",
    "an overly confident person",
    "a pessimist",
    "an optimist",
    "a minimalist",
    "a collector",
    "a tech-savvy person",
    "a conspiracy theorist",
    "a gambler",
    "a diplomat",
    "a retired person",
    "a homeless person",
    "a fashion designer",
    "a dancer",
    "a firefighter",
    "a nurse",
    "a taxi driver",
    "an introvert",
    "an extrovert",
    "a geek",
    "a shy person",
    "a fitness trainer",
    "an undercover agent",
    "a musician",
    "a sailor",
    "an inventor",
    "a dreamer",
    "an entrepreneur",
    "a mathematician",
    "a film director",
    "a puppeteer",
    "an animal lover",
    "a detective novelist",
    "a medieval knight",
    "a vampire",
    "a ghost",
    "a circus performer",
]


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
        json_part = input_string.split("[", 1)[-1].rsplit("]", 1)[0]
        # Replace newlines and extra spaces
        json_cleaned = json_part.replace("\n", "").strip()
        # Load the cleaned JSON content
        return json.loads(f"[{json_cleaned}]")
    except json.JSONDecodeError as e:
        print("Error parsing the input string:", e)
        return []


class OpenAIDatasetGenerator(DatasetGenerator):
    def __init__(self, prompt_template: str) -> None:
        super().__init__(prompt_template)
        self.client = OpenAI(api_key = None)

    def generate(
        self,
        context: str,
        classes: list[str],
        samples_per_class: int = 5,
        number_of_words: int = 50,
        samples_per_inference: int = 10,
    ):
        global who_you_are_options
        responses = defaultdict(list)
        random.shuffle(who_you_are_options)
        print(samples_per_class)

        for specific_class in classes:
            who_you_are_list = who_you_are_options.copy()
            examples = ""
            for _ in tqdm(
                range(int(samples_per_class / samples_per_inference)),
                desc=f"Generating synthetic data for class {specific_class}",
            ):
                who_you_are = random.choice(who_you_are_list)
                prompt = self.prompt_template.format(
                    context=context,
                    specific_class=specific_class,
                    number_of_words=number_of_words,
                    examples=examples,
                    classes=classes,
                    samples_per_inference=samples_per_inference,
                    who_you_are=who_you_are,
                )
                message = [
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": " You are an expert in creating examples for text classification datasets. Your task is to generate highly diverse and unique examples that cover various patterns and nuances to improve the dataset's quality. Ensure that the examples are distinct from each other and encompass a wide range of scenarios relevant to the classification task.",
                            }
                        ],
                    },
                    {"role": "user", "content": [{"type": "text", "text": prompt}]},
                ]
                completion = self.client.chat.completions.create(
                    model="gpt-4o-mini", messages=message
                )

                response = completion.choices[0].message.content
                list_response = parse_to_list(response)
                print(list_response)
                responses[specific_class].extend(list_response)
                for resp in list_response:
                    examples += f"{resp}\n\n"

        return responses
