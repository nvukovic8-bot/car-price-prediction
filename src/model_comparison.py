from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        max_depth=10,
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=5,
        max_depth=10,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42
    )
}

results = []
best_name = None
best_mae = None
best_model = None

for name, estimator in models.items():
    model = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", estimator)
    ])

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    mse = mean_squared_error(y_test, pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, pred)

    results.append({
        "model": name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    })

    if best_mae is None or mae < best_mae:
        best_mae = mae
        best_name = name
        best_model = model

results_df = pd.DataFrame(results).sort_values("MAE")

print(results_df)
print()
print("Najbolji model:", best_name)

best_estimator = best_model.named_steps["model"]

final_model = Pipeline([
    ("preprocessor", build_preprocessor()),
    ("model", best_estimator)
])

final_model.fit(X, y)

joblib.dump(
    final_model,
    MODELS_DIR / "car_price_model.joblib"
)

results_df.to_csv(
    MODELS_DIR / "model_results.csv",
    index=False
)

print("Finalni model sacuvan.")
