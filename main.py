from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
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

# === Эндпоинт API ===
@app.post("/predict")
def predict(input: CommentInput):
    seq = tokenizer.texts_to_sequences([input.text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
    pred = model.predict(padded)[0]
    result = {label: float(pred[i]) for i, label in enumerate(LABELS)}
    return result

# === Веб-интерфейс на корневом маршруте ===
@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Toxic Classifier</title>
        <style>
            body { background: #b2e68d; font-family: sans-serif; padding: 20px; }
            textarea { width: 100%; height: 150px; font-size: 16px; padding: 10px; border-radius: 10px; }
            button { margin-top: 10px; padding: 10px 20px; font-size: 16px; border: none; border-radius: 10px; background: #3498db; color: white; cursor: pointer; }
            .tag { display: inline-block; margin: 5px; padding: 5px 10px; border-radius: 20px; background: #ccc; color: black; font-weight: bold; }
            .active { background: #2ecc71; color: white; }
        </style>
    </head>
    <body>
        <h2>🧠 Multi-label Toxic Comment Classifier</h2>
        <textarea id="text" placeholder="Enter your comment here..."></textarea><br>
        <button onclick="classify()">🔍 Predict</button>

        <div id="results" style="margin-top:20px;"></div>

        <script>
        function classify() {
            const text = document.getElementById("text").value;
            fetch("/predict", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({text: text})
            })
            .then(response => response.json())
            .then(data => {
                let html = "<h3>Predicted Labels:</h3><div>";
                const labels = Object.keys(data);
                labels.forEach(label => {
                    const active = data[label] >= 0.5 ? "active" : "";
                    html += `<span class="tag ${active}">${label}</span>`;
                });
                html += "</div>";
                document.getElementById("results").innerHTML = html;
            });
        }
        </script>
    </body>
    </html>
    """
