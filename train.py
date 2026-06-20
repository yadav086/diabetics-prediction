import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer
from sklearn.impute import SimpleImputer

# Load data
df = pd.read_csv("diabetes.csv")

# Clean zero values
zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[zero_cols] = df[zero_cols].replace(0, np.nan)

# Target and features
X = df[
    [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age",
    ]
]
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.40, random_state=43, stratify=y
)

# Pipeline components
num_columns = X.columns.tolist()
transform_column = Pipeline(
    steps=[
        ("missing", SimpleImputer(strategy="median", missing_values=np.nan)),
        ("Scale", PowerTransformer(method="yeo-johnson")),
    ]
)

preprocess = ColumnTransformer(
    transformers=[("nums", transform_column, num_columns)],
    remainder="passthrough",
)

# Instantiate the model pipeline
model_pipeline = Pipeline(
    steps=[
        ("preprocess", preprocess),
        (
            "log",
            LogisticRegression(
                class_weight="balanced",
                solver="liblinear",
                penalty="l1",
                random_state=43,
            ),
        ),
    ]
)

# Train and export
model_pipeline.fit(X_train, y_train)
joblib.dump(model_pipeline, "model_pipeline.pkl")
print("Production pipeline saved successfully as 'model_pipeline.pkl'!")
