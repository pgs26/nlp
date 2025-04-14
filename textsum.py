import pandas as pd
import torch
import nltk
from datasets import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from nltk.translate.bleu_score import sentence_bleu
import evaluate
import matplotlib.pyplot as plt
#pip install rouge_score
#pip install evaluate
nltk.download("punkt")

# Load CSV
df = pd.read_csv("dummy_summarization_data.csv")

# Load tokenizer and model
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Convert pandas to HF dataset
dataset = Dataset.from_pandas(df)

# Preprocessing function
def preprocess_function(examples):
    inputs = [f"summarize: {text}" for text in examples["article"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(examples["summary"], max_length=64, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Tokenize and split dataset
dataset = dataset.map(preprocess_function, batched=True)
dataset = dataset.train_test_split(test_size=0.3, seed=42)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Training arguments
training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    report_to="none"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer
)

trainer.train()

# Evaluation
sample_input = df.iloc[0]["article"]
input_ids = tokenizer(f"summarize: {sample_input}", return_tensors="pt", max_length=512, truncation=True).input_ids
output_ids = model.generate(input_ids, max_length=64, num_beams=4, early_stopping=True)
summary_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print("🔹Generated Summary:\n", summary_text)
print("🔸Reference Summary:\n", df.iloc[0]["summary"])

# BLEU Score
bleu = sentence_bleu([nltk.word_tokenize(df.iloc[0]["summary"])], nltk.word_tokenize(summary_text))
print(f"\n🟢 BLEU Score: {bleu:.4f}")

# ROUGE Score
rouge = evaluate.load("rouge")
results = rouge.compute(predictions=[summary_text], references=[df.iloc[0]["summary"]])

for k, v in results.items():
    print(f"{k}: {v:.4f}")

plt.bar(results.keys(), results.values(), color="skyblue")
plt.title("ROUGE Scores")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.show()

""" Dataset example 
article	summary
In a major breakthrough, scientists have developed a new material that could revolutionize the energy industry. This material could be used to develop cheaper and more efficient solar panels, reducing global dependence on fossil fuels. The discovery is seen as a major step in the fight against climate change and may transform how we harness energy.	Scientists created a material to improve solar panels and fight climate change.
The city council has approved a new plan to improve public transportation by expanding the metro lines. The goal is to reduce traffic congestion and lower pollution levels by encouraging more people to use public transit.	City council approved a plan to expand metro lines and reduce pollution.
"""
