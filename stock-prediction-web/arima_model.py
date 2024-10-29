import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

def train_arima_model(data):
    # 株価の終値を使用して時系列データを作成
    stock_prices = data['Close']

    # ARIMAモデルのパラメータ (p=5, d=1, q=0) は調整可能
    model = ARIMA(stock_prices, order=(5, 1, 0))
    model_fit = model.fit()

    # 最後の値を予測
    predicted_price = model_fit.forecast()[0]
    
    return predicted_price
