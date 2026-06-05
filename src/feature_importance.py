from pathlib import Path
import pandas as pd
import joblib
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent

df = pd.read_csv(BASE_DIR / "data" / "heart_cleaned.csv")

X = df.drop("target", axis=1)

model = joblib.load(
    BASE_DIR / "models" / "rf_model.pkl"
)

importance = model.feature_importances_

plt.figure(figsize=(10,6))
plt.barh(X.columns, importance)
plt.title("Feature Importance")
plt.tight_layout()

plt.savefig(
    BASE_DIR / "results" / "feature_importance.png"
)

plt.show()