from dotenv import load_dotenv
from src import *
from src.classifier import Classifier, NaiveBayesClassifier, BertClassifier
from pathlib import Path
import json
import logging

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Set variables

# Generate Synthetic Data
generate_synthetic_data = True
input_synthetic_data_path = Path("./results/synthetic_dataset.json") # Only used if generate_synthetic_data is False

# LLM
model_repo = "mistralai/Mixtral-8x7B-Instruct-v0.1"
samples_per_class = 5
number_of_words = 50

# Classifier
labels =["ML", "Testing", "Devops"]
classifier_type = "NB"
context = "Classify jobs descriptions, a job description is a paragraph talking about a job"

# Output
output_dir = Path("./results")

# Example
example = "Lets work with llms, with data analysis, and with neural networks"

def main():
    logging.info("Starting Auto Classifier!\n")
    
    logging.info(f"Classes: {labels}")
    
    if generate_synthetic_data:
        logging.info("Generating Synthetic Data...")
        logging.info(f"- Model repo: {model_repo}")
        logging.info(f"- Samples per Class: {samples_per_class}")
        logging.info(f"- Number of Words: {number_of_words}\n")
        
        # Generate Synthetic Data
        generator = DatasetGenerator(model_repo=model_repo)

        synthetic_data = generator.generate(context=context,
                                    classes=labels,
                                    samples_per_class = samples_per_class,
                                    number_of_words=number_of_words)
        
        # Store the output
        # Create dir if necessary
        output_dir.mkdir(parents=True, exist_ok=True)
        # Writing to a JSON file
        with open(output_dir / "synthetic_dataset.json", "w") as f:
            json.dump(synthetic_data, f, indent=4) 
    else:
        # Load the data
        logging.info(f"Loading data from {input_synthetic_data_path}\n")
        if not input_synthetic_data_path.exists():
            raise FileNotFoundError
        with open(input_synthetic_data_path, "r") as f:
            synthetic_data = json.load(f)
        
    
    # Create a Classifier
    logging.info(f"\nUsing Classifier: {classifier_type}...\n")
    if classifier_type == "NB":
        classifier = NaiveBayesClassifier(labels=labels)
    elif classifier_type == "bert":
        classifier = BertClassifier(labels=labels)

    logging.info("Starting train:... \n")
    # Train the Classifier
    classifier.train(data=synthetic_data)

    # Save the model
    classifier.export(output_dir = output_dir)
    
    # Make a prediction with the example
    prediction = classifier.predict(example)
    

    logging.info("\n=================== EXAMPLE OUTPUT ===================\n")

    
    logging.info(f"Classifier Type: {classifier_type}")
    logging.info(f"Classes: {labels}")
    logging.info(f"Example:'{example}'")
    logging.info("Result: ", prediction)
    
if __name__ == "__main__":
    main()