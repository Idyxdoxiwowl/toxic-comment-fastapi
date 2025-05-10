from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from pathlib import Path
import joblib

# === Константы ===
MODEL_PATH = "logreg_pipeline.pkl"
LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
THRESHOLD = 0.5

# === Загрузка модели ===
pipeline = joblib.load(MODEL_PATH)

# === FastAPI ===
app = FastAPI()

class InputText(BaseModel):
    text: str

@app.post("/predict")
def predict(input: InputText):
    # Прогноз вероятностей
    probs = pipeline.predict_proba([input.text])
    
    result = {
        LABELS[i]: bool(probs[0][i] > THRESHOLD)
        for i in range(len(LABELS))
    }
    return result

@app.get("/", response_class=HTMLResponse)
def serve_index():
    return Path("index.html").read_text(encoding="utf-8")
