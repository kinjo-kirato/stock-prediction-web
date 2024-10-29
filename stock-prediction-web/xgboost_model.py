import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def train_xgboost_model(data):
    # 特徴量とターゲットを分割
    X = data[['Open', 'High', 'Low', 'Volume']].values
    y = data['Close'].values

    # 訓練データとテストデータに分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # XGBoostの回帰モデルを訓練
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000)
    model.fit(X_train, y_train)

    # テストデータでの予測
    predicted_price = model.predict(X_test)

    # MSEで評価
    mse = mean_squared_error(y_test, predicted_price)
    print(f'XGBoost Mean Squared Error: {mse}')

    # 最後のテストデータの予測結果を返す
    return predicted_price[-1]
