from pathlib import Path
import pandas as pd
import joblib
import shap

BASE_DIR = Path(__file__).resolve().parent.parent

df = pd.read_csv(
    BASE_DIR / "data" / "heart_cleaned.csv"
)

X = df.drop("target", axis=1)

model = joblib.load(
    BASE_DIR / "models" / "rf_model.pkl"
)

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X)

shap.summary_plot(
    shap_values,
    X
)