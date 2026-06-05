from pathlib import Path
import pandas as pd
import numpy as np

columns = [
    'age','sex','cp','trestbps','chol',
    'fbs','restecg','thalach','exang',
    'oldpeak','slope','ca','thal','target'
]

BASE_DIR = Path(__file__).resolve().parent.parent

file_path = BASE_DIR / "data" / "processed.cleveland.data"

# Load dataset FIRST
df = pd.read_csv(file_path, names=columns)

# Replace ? with NaN
df.replace('?', np.nan, inplace=True)

# Convert columns to numeric
df['ca'] = pd.to_numeric(df['ca'])
df['thal'] = pd.to_numeric(df['thal'])

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Fill missing values
df.fillna(df.median(numeric_only=True), inplace=True)

# Convert target to binary
df['target'] = df['target'].apply(
    lambda x: 0 if x == 0 else 1
)

print("\nTarget Distribution:")
print(df['target'].value_counts())

# Save cleaned dataset
df.to_csv(
    BASE_DIR / "data" / "heart_cleaned.csv",
    index=False
)

print("\nCleaned dataset saved successfully!")