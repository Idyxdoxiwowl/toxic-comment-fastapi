from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.nn.functional import sigmoid
from fastapi.responses import HTMLResponse
from pathlib import Path

# === Настройки ===
MODEL_PATH = "bert_toxic_model"  # если в папке
LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
THRESHOLD = 0.5

# === Загрузка модели и токенизатора ===
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

# === FastAPI ===
app = FastAPI()

class InputText(BaseModel):
    text: str

@app.post("/predict")
def predict(input: InputText):
    with torch.no_grad():
        tokens = tokenizer(input.text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        logits = model(**tokens).logits
        probs = sigmoid(logits).squeeze().tolist()

        result = {
            LABELS[i]: bool(probs[i] > THRESHOLD)
            for i in range(len(LABELS))
        }
        return result

# === Фронтенд ===
@app.get("/", response_class=HTMLResponse)
def serve_index():
    return Path("index.html").read_text(encoding="utf-8")
