from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "models/performance_classifier.joblib"

st.set_page_config(page_title="Student Performance AI", page_icon="🎓")
st.title("🎓 Student Performance Prediction")
st.write("Predict a student's academic performance category using machine learning.")

if not MODEL_PATH.exists():
    st.warning("Model not found. Run `python src/train.py` first.")
    st.stop()

model = joblib.load(MODEL_PATH)

study = st.slider("Study hours per day", 0.0, 12.0, 5.0)
attendance = st.slider("Attendance (%)", 0.0, 100.0, 80.0)
previous = st.slider("Previous exam score", 0.0, 100.0, 70.0)
assignment = st.slider("Assignment score", 0.0, 100.0, 75.0)
sleep = st.slider("Sleep hours per day", 0.0, 12.0, 7.0)
internet = st.slider("Internet usage hours per day", 0.0, 15.0, 4.0)
activities = st.slider("Extracurricular activities", 0, 10, 2)

if st.button("Predict Performance"):
    X = pd.DataFrame([[
        study, attendance, previous, assignment, sleep, internet, activities
    ]], columns=[
        "study_hours", "attendance_percent", "previous_score",
        "assignment_score", "sleep_hours", "internet_hours",
        "extracurricular_activities"
    ])

    prediction = model.predict(X)[0]
    st.success(f"Predicted Performance: **{prediction}**")

    if hasattr(model, "predict_proba"):
        confidence = model.predict_proba(X).max()
        st.info(f"Model confidence: {confidence:.1%}")

    tips = {
        "Low": "Focus on consistent study, attendance and assignment completion.",
        "Medium": "Maintain your current routine and improve weak areas.",
        "High": "Keep your study habits consistent and continue monitoring performance."
    }
    st.write("### Suggested Focus")
    st.write(tips[prediction])
