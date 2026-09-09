from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def clean_data(df):
    df = df.copy()

    df.columns = [c.strip() for c in df.columns]

    df = df.rename(columns={
        "mileage(kilometers)": "mileage_km",
        "volume(cm3)": "engine_volume_cm3"
    })

    for col in ["priceUSD", "year", "mileage_km", "engine_volume_cm3"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[
        (df["priceUSD"] > 0) &
        (df["year"].between(1900, 2026)) &
        (df["mileage_km"] >= 0)
    ].copy()

    categorical_columns = [
        "make", "model", "condition", "fuel_type",
        "color", "transmission", "drive_unit", "segment"
    ]

    for col in categorical_columns:
        df[col] = df[col].apply(
            lambda x: x.strip().lower() if isinstance(x, str) else np.nan
        )

    return df


if __name__ == "__main__":
    df = pd.read_csv(DATA_DIR / "cars.csv")
    df = clean_data(df)
    df.to_csv(DATA_DIR / "cars_cleaned.csv", index=False)

    print(df.shape)
    print(df.isna().sum())
