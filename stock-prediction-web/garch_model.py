import pandas as pd
from arch import arch_model

def train_garch_model(data):
    # 株価の終値を使用して時系列データを作成
    stock_prices = data['Close']

    # 対数リターンを計算
    returns = 100 * stock_prices.pct_change().dropna()

    # GARCH(1, 1)モデルを作成
    model = arch_model(returns, vol='Garch', p=1, q=1)
    model_fit = model.fit(disp="off")

    # 次のリターンの予測
    predicted_return = model_fit.forecast(horizon=1).mean['h.1'].iloc[-1]

    # 予測されたリターンを使って価格を予測
    last_price = stock_prices.iloc[-1]
    predicted_price = last_price * (1 + predicted_return / 100)
    
    return predicted_price
