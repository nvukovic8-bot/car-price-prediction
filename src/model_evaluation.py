from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_cleaning import clean_data
from feature_engineering import add_features
from data_preprocessing import all_features

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

model = joblib.load(
    MODELS_DIR / "linear_regression_model.joblib"
)

pred = model.predict(X_test)

mae = mean_absolute_error(y_test, pred)
mse = mean_squared_error(y_test, pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, pred)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2:", r2)

examples = pd.DataFrame({
    "actual_price": y_test.values[:10],
    "predicted_price": pred[:10]
})

examples["error"] = (
    examples["actual_price"] -
    examples["predicted_price"]
).abs()

print()
print(examples)
