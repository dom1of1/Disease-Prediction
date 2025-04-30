# README.md
# 🧠 Symptom Checker AI

A lightweight web app that predicts likely health conditions based on user-entered symptoms using a machine learning model.

## Features
- Web-based symptom checker with interactive UI
- Trained Naive Bayes model on real-world health data
- Predicts top 3 likely conditions with confidence scores

## Usage

### 1. Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Train the model:
```bash
python train_model.py
```

### 3. Run the app:
```bash
streamlit run app.py
```

## Dataset
Place the `Training.csv` file from the Kaggle dataset into the `data/` folder. The app uses this file to train and identify symptoms.

Dataset: [Disease Prediction Dataset](https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning)

## Future Improvements
- Integrate more robust ML models (e.g. Random Forest)
- Add advice/treatment recommendations
- Improve UI/UX with advanced filtering and search
