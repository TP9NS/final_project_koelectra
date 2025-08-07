from transformers import ElectraTokenizer, ElectraForSequenceClassification
import torch

# 1. 모델/토크나이저 로드
model = ElectraForSequenceClassification.from_pretrained("saved_model_ver1")
tokenizer = ElectraTokenizer.from_pretrained("saved_model_ver1")

# 2. 예측하고 싶은 문장
question = "문제해결 경험이 있나요?"
answer = "저는 문제를 맞닥들이면 그 문제만 보고 무조건 해결하려고 합니다."

# 3. 전처리
inputs = tokenizer(
    question + " [SEP] " + answer,
    return_tensors="pt",
    truncation=True,
    padding="max_length",
    max_length=256,
)

# 4. 예측
with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(probs, dim=-1).item()

# 5. 결과 출력
labels = ["terminate", "continue"]
print(f"예측 결과: {labels[predicted_class]} ({probs[0][predicted_class]:.2f})")
