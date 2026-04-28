# Document Classification AI

This project classifies business documents such as invoices, contracts, bank statements, and ID documents using Natural Language Processing (NLP) and Machine Learning.

## Tech Stack
- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorization
- FastAPI
- Joblib

## Features
- Text preprocessing
- Document classification using Machine Learning
- TF-IDF feature extraction
- REST API for real-time document prediction
- Clean GitHub-ready folder structure

## Project Structure
```text
document-classification-ai/
├── api/
│   └── app.py
├── data/
│   └── documents.csv
├── models/
├── src/
│   ├── train_model.py
│   └── predict.py
├── requirements.txt
├── .gitignore
└── README.md
```

## How to Run

### 1. Install packages
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python src/train_model.py
```

### 3. Test prediction
```bash
python src/predict.py
```

### 4. Run API
```bash
uvicorn api.app:app --reload
```

Open:
```text
http://127.0.0.1:8000/docs
```


