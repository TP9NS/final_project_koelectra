# train.py

from datasets import load_dataset, DatasetDict, ClassLabel
from transformers import ElectraTokenizer, ElectraForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import evaluate
import os

MODEL_NAME = "monologg/koelectra-small-discriminator"
CSV_PATH = "interview_dataset2.csv"

# 1. Load dataset
dataset = load_dataset("csv", data_files={"train": CSV_PATH}, encoding="utf-8")

# 2. Convert label to ClassLabel (int)
class_names = ["terminate", "continue"]
class_label = ClassLabel(names=class_names)

def encode_labels(example):
    example["label"] = class_label.str2int(example["label"])
    return example

dataset = dataset.map(encode_labels)

# 3. Load tokenizer
tokenizer = ElectraTokenizer.from_pretrained(MODEL_NAME)

# 4. Preprocessing
def preprocess(batch):
    return tokenizer(
        [q + " [SEP] " + a for q, a in zip(batch["question"], batch["answer"])],
        truncation=True,
        padding="max_length",
        max_length=256,
    )

encoded_dataset = dataset.map(preprocess, batched=True)

# 5. Load model
model = ElectraForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# 6. Metrics
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=predictions, references=labels)

# 7. Training
training_args = TrainingArguments(
    output_dir="./checkpoints",
    evaluation_strategy="no",
    per_device_train_batch_size=16,
    num_train_epochs=10,
    save_steps=500,
    save_total_limit=1,
    logging_dir="./logs",
    logging_steps=100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model("./saved_model_ver1")
