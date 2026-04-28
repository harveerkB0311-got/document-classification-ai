from pathlib import Path
import joblib
from fastapi import FastAPI
from pydantic import BaseModel


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "document_classifier.pkl"

app = FastAPI(title="Document Classification AI API")


class DocumentInput(BaseModel):
    text: str


@app.get("/")
def home():
    return {"message": "Document Classification AI API is running"}


@app.post("/predict")
def predict_document(document: DocumentInput):
    model = joblib.load(MODEL_PATH)
    prediction = model.predict([document.text])[0]

    return {
        "document_type": prediction
    }
