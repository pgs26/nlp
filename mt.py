import pandas as pd
from transformers import MarianMTModel, MarianTokenizer, Trainer, TrainingArguments
from nltk.translate.bleu_score import sentence_bleu
from nltk import word_tokenize, download
from datasets import Dataset
import evaluate
import matplotlib.pyplot as plt
#pip install rouge_score
#pip install evaluate
download('punkt')
download('punkt-tab')
# Load dataset
df = pd.read_csv("seq2seq.csv", encoding='latin-1')

# Convert to HuggingFace Dataset
dataset = Dataset.from_pandas(df.rename(columns={"Source": "src", "Target": "tgt"}))

# Load model & tokenizer
model_name = "Helsinki-NLP/opus-mt-en-es"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Tokenization
def preprocess_data(batch):
    source = tokenizer(batch["src"], max_length=128, padding="max_length", truncation=True)
    target = tokenizer(batch["tgt"], max_length=128, padding="max_length", truncation=True)
    source["labels"] = target["input_ids"]
    return source

dataset = dataset.map(preprocess_data, batched=True)
dataset = dataset.train_test_split(test_size=0.2, seed=42)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Training args
training_args = TrainingArguments(
    output_dir="output",
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

# Translate example and evaluate
def translate(text):
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translated = model.generate(**tokens, max_length=100)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

def evaluate_translation(predicted, reference):
    print("\n🔹 Predicted Translation:\n", predicted)
    print("🔸 Reference Translation:\n", reference)

    bleu = sentence_bleu([word_tokenize(reference)], word_tokenize(predicted))
    print(f"\n🟢 BLEU Score: {bleu:.4f}")

    rouge = evaluate.load("rouge")
    scores = rouge.compute(predictions=[predicted], references=[reference])

    print("\n🟢 ROUGE Scores:")
    for k, v in scores.items():
        print(f"{k}: {v:.4f}")

    plt.bar(scores.keys(), scores.values(), color="skyblue")
    plt.title("ROUGE Scores - Machine Translation")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.show()

translated_text = translate("Hello, how are you?")
evaluate_translation(translated_text, "Hola, cómo estás?")

""" lang tags 
English	en
German	de
French	fr
Spanish	es
Italian	it
Portuguese	pt
Romanian	ro
Dutch	nl
Russian	ru
Arabic	ar
Chinese (Simplified)	zh
Japanese	ja
Korean	ko
Hindi	hi

sample data 
Source	Target
hello	hola
good morning	buenos d�as
how are you	c�mo est�s
thank you	gracias"""
