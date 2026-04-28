from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
import json
import re


def load_data(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

train_data = load_data("untitled folder/lab9_dataset/train.json")

# Extract contexts
contexts = [item["context"] for item in train_data]

print(f"\n Loaded {len(contexts)} contexts")


print("\n Loading model...")

model_name = "distilbert-base-cased-distilled-squad"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

print(" Model loaded!")


def find_best_context(question):
    # Better tokenization
    question_words = re.findall(r'\w+', question.lower())

    best_context = ""
    max_match = 0

    for context in contexts:
        match = sum(1 for word in question_words if word in context.lower())

        if match > max_match:
            max_match = match
            best_context = context

    #  Threshold to avoid wrong answers
    if max_match < 2:
        return None

    return best_context


def answer_question(question, context):
    inputs = tokenizer(
        question,
        context,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)

    start = torch.argmax(outputs.start_logits)
    end = torch.argmax(outputs.end_logits) + 1

    answer = tokenizer.decode(inputs["input_ids"][0][start:end])

    return answer.strip()


while True:
    question = input("\n❓ Ask a question (type 'exit' to quit): ")

    if question.lower() == "exit":
        print(" Exiting...")
        break

    # Step 1: Find relevant context
    context = find_best_context(question)

    if context is None:
        print(" No relevant context found")
        continue

    # Step 2: Extract answer
    answer = answer_question(question, context)

    if answer:
        print("\n Answer:", answer)
    else:
        print("\n No answer found")