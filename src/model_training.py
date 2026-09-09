from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

from data_cleaning import clean_data
from feature_engineering import add_features
from data_preprocessing import build_preprocessor, all_features

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

df = pd.read_csv(DATA_DIR / "cars.csv")
df = clean_data(df)
df = add_features(df)

X = df[all_features]
y = df["priceUSD"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = Pipeline([
    ("preprocessor", build_preprocessor()),
    ("model", LinearRegression())
])

model.fit(X_train, y_train)

joblib.dump(
    model,
    MODELS_DIR / "linear_regression_model.joblib"
)

print("Model sacuvan.")
