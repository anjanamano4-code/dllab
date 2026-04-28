from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch


classifier = pipeline('sentiment-analysis')

# Single predictions
print("\n--- Single Predictions ---")
print(classifier('We are very happy to show you the  Transformers library.'))
print(classifier('The pizza is not that great but the crust is awesome.'))

# Batch predictions
print("\n--- Batch Predictions ---")
results = classifier([
    "We are very happy to show you the  Transformers library.",
    "We hope you don't hate it."
])

for result in results:
    print(f"label: {result['label']}, score: {round(result['score'], 4)}")



print("\n--- Custom Model Prediction ---")

model_name = "nlptown/bert-base-multilingual-uncased-sentiment"

model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

print(classifier("I am a good boy"))


print("\n--- Tokenization Example ---")

model_name = "distilbert-base-uncased-finetuned-sst-2-english"

model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

inputs = tokenizer("We are very happy to show you the  Transformers library.")
print(inputs)


# Batch tokenization (PyTorch tensors)
batch = tokenizer(
    [
        "We are very happy to show you the  Transformers library.",
        "We hope you don't hate it."
    ],
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt"
)

for key, value in batch.items():
    print(f"{key}: {value}")

print("\n--- Model Inference ---")

with torch.no_grad():
    outputs = model(**batch)

print(outputs)

# Apply Softmax
probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(probabilities)



print("\n--- Training Example ---")

labels = torch.tensor([1, 0])

outputs_with_loss = model(**batch, labels=labels)
print(outputs_with_loss)

print("\n--- Saving Model ---")

save_directory = "./saved_model"

tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)

print("\n--- Loading Saved Model ---")

tokenizer = AutoTokenizer.from_pretrained(save_directory)
model = AutoModelForSequenceClassification.from_pretrained(save_directory)

print("\n--- Hidden States & Attention ---")

outputs = model(
    **batch,
    output_hidden_states=True,
    output_attentions=True
)

all_hidden_states = outputs.hidden_states
all_attentions = outputs.attentions

print("Hidden states:", len(all_hidden_states))
print("Attention layers:", len(all_attentions))

print("\n=== FINAL PREDICTIONS ===")

classifier = pipeline('sentiment-analysis')

sample_texts = [
    "I love this product, it's amazing!",
    "This is the worst experience ever.",
    "It's okay, not bad but not great."
]

predictions = classifier(sample_texts)

for text, pred in zip(sample_texts, predictions):
    print(f"\nInput: {text}")
    print(f"Prediction: {pred['label']}")
    print(f"Confidence: {round(pred['score'], 4)}")


print("\n--- User Input Prediction ---")

user_input = input("Enter a sentence: ")
result = classifier(user_input)

print("\nPrediction:", result[0]['label'])
print("Confidence:", round(result[0]['score'], 4))