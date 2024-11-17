from dotenv import load_dotenv
from src import *

load_dotenv()

generator = DatasetGenerator(model_repo="mistralai/Mixtral-8x7B-Instruct-v0.1")

output = generator.generate(context="Classify jobs descriptions, a job description is a paragraph talking about a job",
                            classes=["ML", "Testing", "Devops"])

print("=================== OUTPUT ===================")

for specific_class, texts in output.items():
    print(f"- {specific_class} :\n")
    for text in texts:
        print(f"{text}\n")
        
    print("\n\n")