from transformers import ElectraForSequenceClassification, ElectraTokenizer
import torch

model = ElectraForSequenceClassification.from_pretrained("./saved_model_ver1")
tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-small-discriminator")

def predict(question, answer):
    text = question + " [SEP] " + answer
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted = torch.argmax(logits, dim=-1).item()
    return ["terminate", "continue"][predicted]

print(predict("자기소개 해주세요", "음... 그냥 평범하게 살았습니다."))
