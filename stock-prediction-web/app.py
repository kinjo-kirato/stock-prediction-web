import numpy as np
from datetime import datetime, timedelta
import jpholiday  # 日本の祝日を判定するためのライブラリ
from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import yfinance as yf
from lstm_model import train_lstm_model
from q_learning import QLearningAgent, stock_prediction_with_q_learning
from xgboost_model import train_xgboost_model
from arima_model import train_arima_model
from garch_model import train_garch_model
from flask_migrate import Migrate
import matplotlib.pyplot as plt
import io
from flask_cors import CORS
import matplotlib

matplotlib.use('Agg')  # GUIバックエンドを無効化

app = Flask(__name__)
CORS(app)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///stock_data.db'
db = SQLAlchemy(app)
migrate = Migrate(app, db)

# データベースモデルの定義
class StockPrediction(db.Model):
    __tablename__ = 'stock_prediction'
    id = db.Column(db.Integer, primary_key=True)
    execution_date = db.Column(db.Date, nullable=False)  # 実行日
    ticker = db.Column(db.String(10), nullable=False)  # 銘柄コード
    model_type = db.Column(db.String(50), nullable=False)  # 使用した機械学習モデル
    current_price = db.Column(db.Float, nullable=False)  # 実行日の株価
    prediction_date = db.Column(db.Date, nullable=False)  # 予測日
    predicted_price = db.Column(db.Float, nullable=False)  # 予測株価

# 株価データを取得する関数
def fetch_stock_data(ticker):
    data = yf.download(ticker, period='5y', interval='1d')
    return data

# 土日および日本の祝日を考慮して次の平日を取得する関数
def get_next_business_day(start_date, days_ahead):
    target_date = start_date + timedelta(days=days_ahead)
    while target_date.weekday() >= 5 or jpholiday.is_holiday(target_date):
        target_date += timedelta(days=1)
    return target_date

# ストップ高・ストップ安を考慮して予測株価を調整する関数
def apply_price_limits(current_price, predicted_price):
    upper_limit = current_price * 1.3  # ストップ高 30%上限
    lower_limit = current_price * 0.7  # ストップ安 30%下限
    return max(min(predicted_price, upper_limit), lower_limit)

# 予測株価を取得する関数
def get_predicted_prices(stock_symbol):
    # 最新の予測データを取得
    predictions = StockPrediction.query.filter_by(ticker=stock_symbol).all()
    
    if not predictions:
        return None

    # 予測価格のリストを作成
    predicted_prices = [prediction.predicted_price for prediction in predictions]
    
    return predicted_prices

# メインページ
@app.route('/')
def index():
    return render_template('index.html')

# 株価予測API
@app.route('/predict', methods=['POST'])
def predict_stock():
    data = request.json
    ticker = data['ticker']
    model_type = data['model_type']
    period = int(data['period'])

    # 株価データを取得
    stock_data = fetch_stock_data(ticker)
    current_price = stock_data['Close'][-1]

    # 選択されたモデルで予測
    if model_type == 'lstm':
        predicted_price = train_lstm_model(stock_data)
    elif model_type == 'xgboost':
        predicted_price = train_xgboost_model(stock_data)
    elif model_type == 'arima':
        predicted_price = train_arima_model(stock_data)
    elif model_type == 'garch':
        predicted_price = train_garch_model(stock_data)
    else:
        return jsonify({'error': 'Invalid model type selected'}), 400

    # Q-learningを使った強化学習の実行
    q_agent = stock_prediction_with_q_learning(stock_data, predicted_price)

    # 土日・祝日を除く予測日を取得
    execution_date = datetime.now().date()
    prediction_date = get_next_business_day(execution_date, period)

    # ストップ高・ストップ安を考慮した予測値
    adjusted_predicted_price = apply_price_limits(current_price, float(predicted_price))

    # 結果をデータベースに保存
    stock_prediction = StockPrediction(
        execution_date=execution_date,
        ticker=ticker,
        model_type=model_type,
        current_price=current_price,
        prediction_date=prediction_date,
        predicted_price=adjusted_predicted_price
    )
    db.session.add(stock_prediction)
    db.session.commit()

    return jsonify({
        'execution_date': execution_date.strftime('%Y-%m-%d'),
        'ticker': ticker,
        'model_type': model_type,
        'current_price': current_price,
        'prediction_date': prediction_date.strftime('%Y-%m-%d'),
        'predicted_price': adjusted_predicted_price
    })

@app.route('/plot/<string:stock_symbol>', methods=['GET'])
def plot_predicted_prices(stock_symbol):
    try:
        # 予測価格を取得する関数を呼び出す
        predicted_prices = get_predicted_prices(stock_symbol)  
        
        if predicted_prices is None:
            return jsonify({"error": "No predicted prices found!"}), 404
        
        # 予測価格をプロットする処理
        plt.figure(figsize=(10, 5))
        plt.plot(predicted_prices, label='Predicted Price', color='blue')
        plt.title(f'Predicted Prices for {stock_symbol}')
        plt.xlabel('Days')
        plt.ylabel('Price')
        plt.legend()

        # プロット画像を保存
        plt.savefig(f'static/plots/{stock_symbol}_predicted_prices.png')
        plt.close()  # プロットを閉じる

        # プロット画像のURLを返す
        return jsonify({"image_url": f"/static/plots/{stock_symbol}_predicted_prices.png"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # データベースの初期化
        app.run(debug=True, port=5002, host='0.0.0.0')
