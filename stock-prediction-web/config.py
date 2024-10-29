import os

class Config:
    # Flaskのシークレットキー（セキュリティ関連の設定）
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'supersecretkey'

    # SQLAlchemyのデータベースURI（SQLiteを使用）
    SQLALCHEMY_DATABASE_URI = 'sqlite:///stock_data.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # 強化学習用のパラメータ
    ALPHA = 0.1  # 学習率
    GAMMA = 0.9  # 割引率
    EPSILON = 0.1  # 探索率
    NUM_EPISODES = 100  # 強化学習のエピソード数

    # Yahoo Finance APIやその他APIのキー
    YAHOO_FINANCE_API_KEY = os.environ.get('Key') or 'your_api_key_here'    # 自分のキーを入力
