import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 1. 路径设置
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "cars_bev_filtered_clean.csv"
MODEL_PATH = BASE_DIR / "model" / "ev_market_price_model.pkl"
IMPORTANCE_PATH = BASE_DIR / "model" / "market_feature_importance.csv"


# 2. 读取数据
df = pd.read_csv(DATA_PATH)

print("数据大小:", df.shape)
print("列名:")
print(df.columns.tolist())


# 3. 选择用于训练的字段
features = [
    "brand",
    "model",
    "model_year",
    "mileage_mi",
    "transmission",
    "drivetrain",
    "accidents_or_damage",
    "one_owner",
    "personal_use_only"
]

target = "price_usd"

df = df[features + [target]].copy()


# 4. 基础清洗
df = df.dropna(subset=[target])
df = df[df[target] > 1000]
df = df[df[target] < 250000]

df = df.dropna(subset=["brand", "model", "model_year", "mileage_mi"])

# 数值型字段
numeric_features = [
    "model_year",
    "mileage_mi",
    "accidents_or_damage",
    "one_owner",
    "personal_use_only"
]

# 类别型字段
categorical_features = [
    "brand",
    "model",
    "transmission",
    "drivetrain"
]

# 缺失值处理
for col in numeric_features:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col] = df[col].fillna(df[col].median())

for col in categorical_features:
    df[col] = df[col].astype(str).fillna("Unknown")


X = df[features]
y = df[target]

print("清洗后数据大小:", X.shape)


# 5. One-Hot Encoding
try:
    onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    onehot = OneHotEncoder(handle_unknown="ignore", sparse=False)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features),
        ("cat", onehot, categorical_features)
    ]
)


# 6. 建立 Random Forest 模型
rf_model = RandomForestRegressor(
    n_estimators=150,
    max_depth=25,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("regressor", rf_model)
    ]
)


# 7. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# 8. 训练模型
print("模型训练中，请稍等...")
model.fit(X_train, y_train)


# 9. 预测和评价
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("====== 新二手电动车价格预测模型结果 ======")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2:", round(r2, 4))


# 10. 保存模型
joblib.dump(model, MODEL_PATH)
print("模型已保存到:", MODEL_PATH)


# 11. 保存特征重要性
feature_names = model.named_steps["preprocess"].get_feature_names_out()
importances = model.named_steps["regressor"].feature_importances_

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values(by="importance", ascending=False)

importance_df.to_csv(IMPORTANCE_PATH, index=False)

print("特征重要性已保存到:", IMPORTANCE_PATH)
print("前15个重要特征:")
print(importance_df.head(15))