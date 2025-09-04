from datasets import Dataset, DatasetDict, ClassLabel
from transformers import ElectraTokenizer, ElectraForSequenceClassification, TrainingArguments, Trainer
import pandas as pd
import numpy as np
import evaluate
import random
import os
import torch
import matplotlib.pyplot as plt

# --------------------
# Config
# --------------------
MODEL_NAME = "monologg/koelectra-small-discriminator"
CSV_PATH = "interview_dataset4.csv"  # ← 원본 CSV (깨진 줄은 스킵 처리)
SEED = 42  # 재현성

# 재현성 고정
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# --------------------
# 1) Load dataset (robust pandas → HF Dataset)
# --------------------
#   - on_bad_lines='skip': 필드 수 안 맞는 줄은 건너뜀
#   - engine='python'    : 따옴표/쉼표 까다로운 케이스에 관대
#   - header=None        : 파일에 헤더가 없다는 가정
#     (만약 첫 줄이 "question,answer,label" 헤더라면 header=0 로 바꾸세요)
df = pd.read_csv(
    CSV_PATH,
    header=None,
    names=["question", "answer", "label"],
    encoding="utf-8",
    on_bad_lines="skip",
    engine="python",
)

# 라벨 정리 및 결측 제거
df = df[df["label"].isin(["terminate", "continue"])].dropna(subset=["question", "answer", "label"])

# HuggingFace Dataset 변환
raw = Dataset.from_pandas(df, preserve_index=False)

# --------------------
# 2) Convert label -> int
# --------------------
class_names = ["terminate", "continue"]
class_label = ClassLabel(names=class_names)

def encode_labels(example):
    # label이 이미 int면 그대로, 아니면 문자열→int
    lbl = example["label"]
    if isinstance(lbl, str):
        example["label"] = class_label.str2int(lbl)
    elif isinstance(lbl, (int, np.integer)):
        example["label"] = int(lbl)
    else:
        example["label"] = class_label.str2int(str(lbl))
    return example

raw = raw.map(encode_labels)

# --------------------
# 3) 9:1 split (stratified)
# --------------------
try:
    split = raw.train_test_split(
        test_size=0.1,
        seed=SEED,
        stratify_by_column="label"
    )
except Exception:
    split = raw.train_test_split(test_size=0.1, seed=SEED)

dataset = DatasetDict({
    "train": split["train"],
    "eval": split["test"],
})

# --------------------
# 4) Tokenizer
# --------------------
tokenizer = ElectraTokenizer.from_pretrained(MODEL_NAME)

def preprocess(batch):
    # question + [SEP] + answer 형태로 하나의 입력 시퀀스 구성
    texts = [q + " [SEP] " + a for q, a in zip(batch["question"], batch["answer"])]
    enc = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=256,
    )
    return enc

# label은 유지하고 나머지 원본 컬럼(question, answer 등)은 제거
remove_cols = [c for c in dataset["train"].column_names if c not in ["label"]]
encoded = dataset.map(preprocess, batched=True, remove_columns=remove_cols)

# --------------------
# 5) Model
# --------------------
model = ElectraForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# --------------------
# 6) Metrics
# --------------------
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    out = {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "f1": f1.compute(predictions=preds, references=labels, average="binary", pos_label=1)["f1"],
    }
    return out

# --------------------
# 7) Training args
# --------------------
training_args = TrainingArguments(
    output_dir="./checkpoints",
    evaluation_strategy="steps",   # ← eval_strategy(X)
    eval_steps=300,
    logging_steps=25,
    save_steps=300,
    save_total_limit=2,
    per_device_train_batch_size=16,
    num_train_epochs=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    logging_dir="./logs",
    seed=SEED,
)

# --------------------
# 8) Trainer
# --------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded["train"],
    eval_dataset=encoded["eval"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model("./saved_model_ver2")

# --------------------
# 9) Logging → Pretty Plots (Train/Eval)
# --------------------
def smooth_ema(values, beta=0.9):
    """Exponential Moving Average with bias correction."""
    if not values:
        return []
    ema, out = 0.0, []
    for i, v in enumerate(values, start=1):
        ema = beta * ema + (1 - beta) * v
        out.append(ema / (1 - beta ** i))
    return out

history = trainer.state.log_history

steps, train_loss, lrs = [], [], []
eval_steps, eval_loss, eval_acc = [], [], []

for h in history:
    step = h.get("step", None)
    if step is None:
        continue

    if "loss" in h:
        steps.append(step); train_loss.append(h["loss"])
    if "learning_rate" in h:
        lrs.append((step, h["learning_rate"]))

    if ("eval_loss" in h) or ("eval_accuracy" in h):
        eval_steps.append(step)
        if "eval_loss" in h: eval_loss.append(h["eval_loss"])
        if "eval_accuracy" in h: eval_acc.append(h["eval_accuracy"])

def pretty_ax(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=16, pad=10)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, which="both", linestyle="-", alpha=0.25)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="both", labelsize=10)

# 1) Train loss
if train_loss:
    fig, ax = plt.subplots(figsize=(12, 4), dpi=300)
    ax.plot(steps, train_loss, linewidth=1.5, alpha=0.35, label="train loss (raw)")
    ax.plot(steps, smooth_ema(train_loss, beta=0.9), linewidth=2.5, label="train loss (EMA)")
    pretty_ax(ax, "Train Loss", "Step", "Loss")
    ax.legend(frameon=False, fontsize=10, loc="best")
    plt.tight_layout(); plt.savefig("logs_train_loss.png", bbox_inches="tight"); plt.close(fig)

# 2) Eval loss
if eval_loss:
    fig, ax = plt.subplots(figsize=(12, 4), dpi=300)
    ax.plot(eval_steps, eval_loss, linewidth=1.8, alpha=0.5, label="eval loss (raw)")
    ax.plot(eval_steps, smooth_ema(eval_loss, beta=0.8), linewidth=2.6, label="eval loss (EMA)")
    pretty_ax(ax, "Eval Loss", "Step", "Loss")
    ax.legend(frameon=False, fontsize=10, loc="best")
    plt.tight_layout(); plt.savefig("logs_eval_loss.png", bbox_inches="tight"); plt.close(fig)

# 3) Eval accuracy
if eval_acc:
    fig, ax = plt.subplots(figsize=(12, 4), dpi=300)
    ax.plot(eval_steps, eval_acc, linewidth=1.8, alpha=0.5, label="eval acc (raw)")
    ax.plot(eval_steps, smooth_ema(eval_acc, beta=0.8), linewidth=2.6, label="eval acc (EMA)")
    pretty_ax(ax, "Eval Accuracy", "Step", "Accuracy")
    ymin = max(0.0, min(eval_acc) - 0.02); ymax = min(1.0, max(eval_acc) + 0.02)
    ax.set_ylim(ymin, ymax)
    ax.legend(frameon=False, fontsize=10, loc="best")
    plt.tight_layout(); plt.savefig("logs_eval_accuracy.png", bbox_inches="tight"); plt.close(fig)

# 4) Learning rate
if lrs:
    lr_steps, lr_vals = zip(*lrs)
    fig, ax = plt.subplots(figsize=(12, 3.8), dpi=300)
    ax.plot(lr_steps, lr_vals, linewidth=2.2)
    pretty_ax(ax, "Learning Rate", "Step", "LR")
    plt.tight_layout(); plt.savefig("logs_learning_rate.png", bbox_inches="tight"); plt.close(fig)

print("Done. Saved model to ./saved_model_ver2 and plots to logs_*.png")
