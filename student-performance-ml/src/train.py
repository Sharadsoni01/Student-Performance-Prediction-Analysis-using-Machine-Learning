from pathlib import Path
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data" / "student_performance_demo.csv"
MODEL_DIR = BASE / "models"
MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA)
features = [
    "study_hours", "attendance_percent", "previous_score",
    "assignment_score", "sleep_hours", "internet_hours",
    "extracurricular_activities"
]

X = df[features]
y_cls = df["performance"]
y_reg = df["final_score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y_cls, test_size=0.20, random_state=42, stratify=y_cls
)

classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=250, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

best_name, best_model, best_acc = None, None, -1

print("CLASSIFICATION RESULTS")
print("-" * 50)
for name, estimator in classifiers.items():
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", estimator)
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print(f"{name}: accuracy = {acc:.4f}")
    if acc > best_acc:
        best_name, best_model, best_acc = name, model, acc

joblib.dump(best_model, MODEL_DIR / "performance_classifier.joblib")

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X, y_reg, test_size=0.20, random_state=42
)

regressors = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=250, random_state=42)
}

print("\nREGRESSION RESULTS")
print("-" * 50)
for name, estimator in regressors.items():
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", estimator)
    ])
    model.fit(Xr_train, yr_train)
    pred = model.predict(Xr_test)
    rmse = mean_squared_error(yr_test, pred) ** 0.5
    print(f"{name}: MAE={mean_absolute_error(yr_test, pred):.4f}, RMSE={rmse:.4f}, R2={r2_score(yr_test, pred):.4f}")

print(f"\nBest classifier: {best_name} | Accuracy: {best_acc:.4f}")
print("Saved models/performance_classifier.joblib")
