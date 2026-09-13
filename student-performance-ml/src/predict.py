from pathlib import Path
import joblib
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
MODEL = BASE / "models/performance_classifier.joblib"

def predict_performance(study_hours, attendance_percent, previous_score,
                        assignment_score, sleep_hours, internet_hours,
                        extracurricular_activities):
    model = joblib.load(MODEL)
    X = pd.DataFrame([[
        study_hours, attendance_percent, previous_score,
        assignment_score, sleep_hours, internet_hours,
        extracurricular_activities
    ]], columns=[
        "study_hours", "attendance_percent", "previous_score",
        "assignment_score", "sleep_hours", "internet_hours",
        "extracurricular_activities"
    ])
    prediction = model.predict(X)[0]
    confidence = None
    if hasattr(model, "predict_proba"):
        confidence = float(model.predict_proba(X).max())
    return prediction, confidence
