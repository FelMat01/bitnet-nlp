from pathlib import Path
import json

from dotenv import load_dotenv

from src import *

load_dotenv()

# Set variables

# Generate Synthetic Data
generate_synthetic_data = True
input_synthetic_data_path = Path("./results/synthetic_dataset.json") # Only used if generate_synthetic_data is False

# LLM
model_repo = "mistralai/Mixtral-8x7B-Instruct-v0.1"
samples_per_class = 5
number_of_words = 50

# Classifier
classifier_type = "NB"
context = "Classify jobs descriptions, a job description is a paragraph talking about a job"
labels = ["ML", "Testing", "Devops"]

# Output
output_dir = Path("./results")

# Example
example = "Lets work with llms, with data analysis, and with predictive models"

def main():
    print("Starting Auto Classifier!")
    
    print(f"- Context: {context}")
    print(f"- Classes: {labels}\n")
    
    if generate_synthetic_data:
        print("Generating Synthetic Data...")
        print(f"- Model repo: {model_repo}")
        print(f"- Samples per Class: {samples_per_class}")
        print(f"- Number of Words: {number_of_words}\n")
        
        # Generate Synthetic Data
        generator = DatasetGenerator(model_repo=model_repo)
        synthetic_data = generator.generate(context=context,
                                    classes=labels,
                                    samples_per_class = samples_per_class,
                                    number_of_words=number_of_words)
        
        # Store the output
        print(f"Storing data in {input_synthetic_data_path}\n")
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(input_synthetic_data_path, "w") as f:
            json.dump(synthetic_data, f, indent=4) 
    else:
        # Load the data
        print(f"Loading data from {input_synthetic_data_path}\n")
        if not input_synthetic_data_path.exists():
            raise FileNotFoundError
        with open(input_synthetic_data_path, "r") as f:
            synthetic_data = json.load(f)
        
    
    # Create a Classifier
    print(f"Using classifier: {classifier_type}")
    if classifier_type == "NB":
        classifier = NaiveBayesClassifier(labels=labels)
    elif classifier_type == "bert":
        classifier = BertClassifier(labels=labels)
    else:
        raise ValueError(f"Unknown classifier: {classifier_type}")

    print("Classifier Train: Starting...")
    # Train the Classifier
    classifier.train(data=synthetic_data)
    print("Classifier Train: OK \n")

    # Save the model
    classifier.export(output_dir = output_dir)
    
    # Make a prediction with the example
    prediction = classifier.predict(example)
    

    print("\n=================== EXAMPLE OUTPUT ===================\n")

    
    print(f"Classifier Type: {classifier_type}")
    print(f"Classes: {labels}")
    print(f"Example:'{example}'")
    print("Result: ", prediction)
    
if __name__ == "__main__":
    main()