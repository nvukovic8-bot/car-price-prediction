from pathlib import Path
import sys
import pandas as pd
import joblib

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR / "src"))

from feature_engineering import add_features
from data_preprocessing import all_features

model = joblib.load(
    BASE_DIR / "models" / "car_price_model.joblib"
)

data = {
    "make": [input("Make: ").strip().lower()],
    "model": [input("Model: ").strip().lower()],
    "year": [int(input("Year: "))],
    "condition": [input("Condition: ").strip().lower()],
    "mileage_km": [float(input("Mileage km: "))],
    "fuel_type": [input("Fuel type: ").strip().lower()],
    "engine_volume_cm3": [float(input("Engine volume cm3: "))],
    "color": [input("Color: ").strip().lower()],
    "transmission": [input("Transmission: ").strip().lower()],
    "drive_unit": [input("Drive unit: ").strip().lower()],
    "segment": [input("Segment: ").strip().lower()]
}

df = pd.DataFrame(data)
df = add_features(df)

prediction = model.predict(df[all_features])[0]

print()
print("Predvidjena cijena: $", round(prediction, 2))
