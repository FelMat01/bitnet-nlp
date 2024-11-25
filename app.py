import json
import pandas as pd
from src import NaiveBayesClassifier, BertClassifier, HFDatasetGenerator, OpenAIDatasetGenerator, KNNClassifier
from datasets import Dataset
from config import SYNTHETIC_DATASET_GENERATE, SYNTHETIC_DATASET_PATH, SYNTHETIC_DATASET_FOLDER, MODELS_FOLDER

# LLM
model_repo = "mistralai/Mixtral-8x7B-Instruct-v0.1"
samples_per_class = 20
number_of_words = 50

# Classifier
classifier_type = "bert"
context = """
Generate text that provides an overview of a role or position, highlighting its responsibilities, qualifications, etc. 
Do it in the tone of something that would be found in a CV, as if you were looking to be hired for a job in that field.
"""
labels = ["ML", "Testing", "Devops"]

# Example
example = "Lets work with llms, with data analysis, and with predictive models"

dataset_generator_type = "HF"#"OpenAI"  "HF"
classifier_class = {"NB" : NaiveBayesClassifier ,"bert" : BertClassifier, "knnEmbeddings" : KNNClassifier }
def main():
    print("Starting Auto Classifier!")
    
    print(f"- Context: {context}")
    print(f"- Classes: {labels}\n")
    
    if SYNTHETIC_DATASET_GENERATE:
        print("Generating Synthetic Data...")
        print(f"- Model repo: {model_repo}")
        print(f"- Samples per Class: {samples_per_class}")
        print(f"- Number of Words: {number_of_words}\n")
        
        # Choose generator
        if dataset_generator_type == "HF":
            dataset_generator = HFDatasetGenerator
        elif dataset_generator_type == "OpenAI":
            dataset_generator = OpenAIDatasetGenerator
        else:
            raise ValueError(f"Unknown Dataset Generator Type: {dataset_generator_type}")
            
        # Generate Synthetic Data
        generator = dataset_generator(model_repo=model_repo)
        synthetic_data = generator.generate(context=context,
                                    classes=labels,
                                    samples_per_class = samples_per_class,
                                    number_of_words=number_of_words)
        
        # Store the output
        print(f"Storing data in {SYNTHETIC_DATASET_PATH}\n")
        SYNTHETIC_DATASET_FOLDER.mkdir(parents=True, exist_ok=True)
        with open(SYNTHETIC_DATASET_PATH, "w") as f:
            json.dump(synthetic_data, f, indent=4) 
    else:
        # Load the data
        print(f"Loading data from {SYNTHETIC_DATASET_PATH}\n")
        if not SYNTHETIC_DATASET_PATH.exists():
            raise FileNotFoundError
        with open(SYNTHETIC_DATASET_PATH, "r") as f:
            synthetic_data = json.load(f)
        


    # Create a Classifier
    print(f"Using classifier: {classifier_type}")

    classifier = classifier_class[classifier_type](labels=labels)


    # Train the Classifier
    print("Classifier Train: Starting...")
    classifier.train(data=synthetic_data)
    print("Classifier Train: OK \n")

    # Save the model
    classifier.export(dir=MODELS_FOLDER)
    
    # Make a prediction with the example
    prediction = classifier.predict(example)
    

    print("\n=================== EXAMPLE OUTPUT ===================\n")

    
    print(f"Classifier Type: {classifier_type}")
    print(f"Classes: {labels}")
    print(f"Example:'{example}'")
    print("Result: ", prediction)
    
if __name__ == "__main__":
    main()