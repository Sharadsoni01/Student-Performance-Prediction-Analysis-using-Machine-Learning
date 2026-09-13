from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

BASE = Path(__file__).resolve().parent.parent
df = pd.read_csv(BASE / "data/student_performance_demo.csv")
features = ["study_hours","attendance_percent","previous_score","assignment_score",
            "sleep_hours","internet_hours","extracurricular_activities"]

X_train, X_test, y_train, y_test = train_test_split(
    df[features], df["performance"], test_size=0.20,
    random_state=42, stratify=df["performance"]
)

model = joblib.load(BASE / "models/performance_classifier.joblib")
pred = model.predict(X_test)

print(classification_report(y_test, pred, zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(y_test, pred))
