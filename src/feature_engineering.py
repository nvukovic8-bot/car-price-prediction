from pathlib import Path
import pandas as pd

from data_cleaning import clean_data

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def add_features(df):
    df = df.copy()

    df["car_age"] = (2026 - df["year"]).clip(lower=1)
    df["mileage_per_year"] = df["mileage_km"] / df["car_age"]
    df["engine_volume_liters"] = df["engine_volume_cm3"] / 1000
    df["is_newer_car"] = (df["year"] >= 2015).astype(int)
    df["is_high_mileage"] = (df["mileage_km"] >= 200000).astype(int)

    df["brand_model"] = (
        df["make"].fillna("unknown") +
        "_" +
        df["model"].fillna("unknown")
    )

    return df


if __name__ == "__main__":
    df = pd.read_csv(DATA_DIR / "cars.csv")
    df = clean_data(df)
    df = add_features(df)

    df.to_csv(DATA_DIR / "cars_features.csv", index=False)

    print(df.shape)
    print(df.head())
