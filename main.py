from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

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

# === Эндпоинт ===
@app.post("/predict")
def predict(input: CommentInput):
    seq = tokenizer.texts_to_sequences([input.text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
    pred = model.predict(padded)[0]

    # Преобразуем в словарь {"label": вероятность}
    result = {label: float(pred[i]) for i, label in enumerate(LABELS)}
    return result


@app.get("/")
def root():
    return {
        "message": "🚀 Добро пожаловать в Toxic Comment Classifier API!",
        "docs": "Откройте /docs для Swagger UI",
        "example": {
            "text": "You're such an idiot."
        },
        "predict_endpoint": "/predict (POST)"
    }
