import joblib
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "document_classifier.pkl"


def predict_document():
    model = joblib.load(MODEL_PATH)

    sample_text = "This invoice includes total amount, tax, billing address, and payment due date."
    prediction = model.predict([sample_text])[0]

    print(f"Document Prediction: {prediction}")


if __name__ == "__main__":
    predict_document()
