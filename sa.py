#pip install datasets
import pandas as pd
from datasets import Dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, TrainingArguments, Trainer, TextClassificationPipeline
from sklearn.preprocessing import LabelEncoder
import torch

# Load and encode dataset
df = pd.read_csv("sentiment.txt")
le = LabelEncoder()
df["label"] = le.fit_transform(df["label"])  # positive=1, negative=0

dataset = Dataset.from_pandas(df)

# Load tokenizer and tokenize dataset
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

dataset = dataset.map(tokenize, batched=True)
dataset = dataset.train_test_split(test_size=0.2)

# Load model
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Define training args
training_args = TrainingArguments(
    output_dir="output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    evaluation_strategy="epoch",
    save_strategy="no",
    logging_strategy="epoch",
    report_to="none"  # Disable WandB logging
)


# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer
)

# Fine-tune
trainer.train()

pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, top_k=None)

print(pipe("I love this!"))
print(pipe("This is bad."))

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# 🧪 Get true labels and predicted labels
test_dataset = dataset["test"]
true_labels = test_dataset["label"]

# 🔍 Get model predictions on test set
predictions = trainer.predict(test_dataset)
preds = np.argmax(predictions.predictions, axis=1)

# ✅ Accuracy and Classification Report
accuracy = accuracy_score(true_labels, preds)
print(f"\n🟢 Accuracy: {accuracy:.4f}\n")
print("🟢 Classification Report:")
print(classification_report(true_labels, preds, target_names=le.classes_))

# 📊 Confusion Matrix
cm = confusion_matrix(true_labels, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - Sentiment Classification")
plt.show()

""" Data Format
text,label
"I love this product",positive
"This is terrible",negative
"""
