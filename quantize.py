import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM

from bitnet import replace_linears_in_hf

# Load a model from Hugging Face's Transformers
model_name = "Qwen/Qwen2.5-1.5B" # GOOD well done <3 8==D O: :D :D
# model_name = "datificate/gpt2-small-spanish"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
# model = model.to('cpu')
# model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Replace Linear layers with BitLinear
# replace_linears_in_hf(model)

# Example text to classify
text = "Es 2+2 igual a 4?"
input_ids = tokenizer.encode(text, return_tensors="pt")
output = model.generate(input_ids, max_length=50)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(generated_text)

# inputs = tokenizer(
#     text, return_tensors="pt", padding=True, truncation=True, max_length=512
# )

# # Perform inference 
# model.eval()  # Set the model to evaluation mode
# with torch.no_grad():
#     outputs = model(**inputs)
#     predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
#     print(predictions)

# # Process predictions
# predicted_class_id = predictions.argmax().item()
# print(f"Predicted class ID: {predicted_class_id}")

# # Optionally, map the predicted class ID to a label, if you know the classification labels
# labels = ["Label 1", "Label 2", ...]  # Define your labels corresponding to the model's classes
# print(f"Predicted label: {labels[predicted_class_id]}")