import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "ev_market_price_model.pkl"

model = joblib.load(MODEL_PATH)

sample = pd.DataFrame({
    "brand": ["Tesla"],
    "model": ["Model 3 Long Range"],
    "model_year": [2022],
    "mileage_mi": [30000],
    "transmission": ["1-Speed Automatic"],
    "drivetrain": ["All-Wheel Drive"],
    "accidents_or_damage": [0],
    "one_owner": [1],
    "personal_use_only": [1]
})

price = model.predict(sample)[0]

print("预测价格:", round(price, 2), "USD")