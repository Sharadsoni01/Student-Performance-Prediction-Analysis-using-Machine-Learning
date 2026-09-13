# Student Performance Prediction & Analysis using Machine Learning

An end-to-end machine learning project that analyzes student-related factors and predicts academic performance.

## Project Goals
- Explore relationships between study habits, attendance, previous scores, assignments, sleep, internet usage and academic performance.
- Preprocess and validate the data.
- Compare multiple machine learning models.
- Predict final performance category: Low, Medium or High.
- Provide an interactive prediction interface.

## ML Workflow

Data → EDA → Train/Test Split → Preprocessing → Model Training → Evaluation → Best Model → Prediction

## Models
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

## Evaluation
Classification models are compared using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Regression is also demonstrated for predicting the numerical final score using:
- Linear Regression
- Random Forest Regressor
- MAE
- RMSE
- R²

## Features
- Study hours
- Attendance percentage
- Previous exam score
- Assignment score
- Sleep hours
- Internet usage hours
- Extracurricular activities

## Dataset
`data/student_performance_demo.csv` is a **synthetic demonstration dataset generated for this project**. It is included so the project runs immediately. Its results must not be presented as real-world research findings.

For a stronger portfolio version, replace it with a documented real student-performance dataset and rerun the full analysis.

## Run

```bash
pip install -r requirements.txt
python src/train.py
streamlit run app.py
```

## Project Structure

```text
student-performance-ml/
├── data/
│   └── student_performance_demo.csv
├── models/
├── notebooks/
├── src/
│   ├── train.py
│   ├── predict.py
│   └── evaluate.py
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```
