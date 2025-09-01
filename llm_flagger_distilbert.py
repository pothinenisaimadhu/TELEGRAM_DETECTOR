import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
import numpy as np
import torch

# ==================================================
# 1. Load your synthetic dataset
# ==================================================
df = pd.read_csv("synthetic_text_classification.csv")  # file with columns: text, label

# Map string labels -> numeric
label2id = {"illegal": 0, "abuse": 1, "adult content": 2, "safe": 3}
id2label = {v: k for k, v in label2id.items()}
df["label"] = df["label"].map(label2id)

# Hugging Face dataset
dataset = Dataset.from_pandas(df)

# Train/test split (80/20)
dataset = dataset.train_test_split(test_size=0.2, seed=42)
train_ds, test_ds = dataset["train"], dataset["test"]

# ==================================================
# 2. Load tokenizer & model
# ==================================================
model_name = "distilbert-base-uncased"  # you can also try "facebook/bart-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_fn(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

train_ds = train_ds.map(tokenize_fn, batched=True)
test_ds = test_ds.map(tokenize_fn, batched=True)

train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Model
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=4,
    id2label=id2label,
    label2id=label2id,
)

# ==================================================
# 3. Metrics
# ==================================================
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy.compute(predictions=preds, references=labels)
    f1_macro = f1.compute(predictions=preds, references=labels, average="macro")
    return {"accuracy": acc["accuracy"], "f1": f1_macro["f1"]}

# ==================================================
# 4. Training
# ==================================================
training_args = TrainingArguments(
    output_dir="./model_output",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
    report_to="none"  # 🚫 disables wandb
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

# ==================================================
# 5. Save final model
# ==================================================
trainer.save_model("./model_output")
tokenizer.save_pretrained("./model_output")

print("✅ Training complete! Model saved at ./model_output")
