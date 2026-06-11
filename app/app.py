from flask import Flask, render_template, request
import joblib
import pandas as pd
from pathlib import Path

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "ev_market_price_model.pkl"
IMPORTANCE_PATH = BASE_DIR / "model" / "market_feature_importance.csv"

model = joblib.load(MODEL_PATH)


def load_feature_importance():
    if not IMPORTANCE_PATH.exists():
        return [
            {"label": "연식", "percent": 30},
            {"label": "주행거리", "percent": 25},
            {"label": "브랜드", "percent": 20},
            {"label": "구동 방식", "percent": 15},
            {"label": "사고 이력", "percent": 10},
        ]

    importance_df = pd.read_csv(IMPORTANCE_PATH)

    grouped = {}

    for _, row in importance_df.iterrows():
        feature = str(row["feature"])
        importance = float(row["importance"])

        if "model_year" in feature:
            key = "연식"
        elif "mileage_mi" in feature:
            key = "주행거리"
        elif "brand" in feature:
            key = "브랜드"
        elif "model" in feature:
            key = "모델"
        elif "drivetrain" in feature:
            key = "구동 방식"
        elif "transmission" in feature:
            key = "변속기"
        elif "accidents_or_damage" in feature:
            key = "사고 이력"
        elif "one_owner" in feature:
            key = "1인 소유"
        elif "personal_use_only" in feature:
            key = "개인 사용"
        else:
            key = "기타"

        grouped[key] = grouped.get(key, 0) + importance

    total = sum(grouped.values())

    items = []
    for key, value in grouped.items():
        percent = round((value / total) * 100, 1)
        items.append({
            "label": key,
            "percent": percent
        })

    items = sorted(items, key=lambda x: x["percent"], reverse=True)

    return items[:5]


importance_items = load_feature_importance()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    brand = request.form["brand"]
    car_model = request.form["model"]
    model_year = int(request.form["model_year"])

    mileage_km = float(request.form["mileage_km"])
    mileage_mi = mileage_km / 1.60934

    transmission = request.form["transmission"]
    drivetrain = request.form["drivetrain"]

    accidents_or_damage = int(request.form["accidents_or_damage"])
    one_owner = int(request.form["one_owner"])
    personal_use_only = int(request.form["personal_use_only"])

    input_data = pd.DataFrame({
        "brand": [brand],
        "model": [car_model],
        "model_year": [model_year],
        "mileage_mi": [mileage_mi],
        "transmission": [transmission],
        "drivetrain": [drivetrain],
        "accidents_or_damage": [accidents_or_damage],
        "one_owner": [one_owner],
        "personal_use_only": [personal_use_only]
    })

    prediction = model.predict(input_data)[0]

    return render_template(
        "result.html",
        prediction=round(prediction, 2),
        brand=brand,
        car_model=car_model,
        model_year=model_year,
        mileage_km=mileage_km,
        mileage_mi=mileage_mi,
        importance_items=importance_items
    )


if __name__ == "__main__":
    app.run(debug=True)