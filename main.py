from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from pathlib import Path

# === Загрузка модели и токенизатора ===
model = tf.keras.models.load_model("lstm_model.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

MAX_LEN = 200
LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# === FastAPI ===
app = FastAPI(title="Toxic Comment Classifier")

# === Модель запроса ===
class CommentInput(BaseModel):
    text: str

# === Предсказания ===
@app.post("/predict")
def predict(input: CommentInput):
    seq = tokenizer.texts_to_sequences([input.text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
    pred = model.predict(padded)[0]
    result = {label: float(pred[i]) for i, label in enumerate(LABELS)}
    return result

# === Роут для отображения index.html ===
@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    return Path("index.html").read_text(encoding="utf-8")
