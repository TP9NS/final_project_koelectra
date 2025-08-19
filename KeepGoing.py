from transformers import ElectraTokenizer, ElectraForSequenceClassification
import torch

# 모델/토크나이저 전역 로드 (최초 1회)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ElectraForSequenceClassification.from_pretrained("saved_model_ver1").to(device)
tokenizer = ElectraTokenizer.from_pretrained("saved_model_ver1")

labels = ["terminate", "continue"]

def keepgoing(question: str, answer: str) -> str:
    inputs = tokenizer(
        question + " [SEP] " + answer,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=256,
    ).to(device)

    # 예측
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(probs, dim=-1).item()

    return labels[predicted_class]
