import numpy as np
import random
import pandas as pd

# Qテーブルの初期化とエージェントの定義
class QLearningAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            action = random.choice(self.actions)
        else:
            state_actions = self.q_table.get(state, {})
            if not state_actions:
                action = random.choice(self.actions)
            else:
                action = max(state_actions, key=state_actions.get)
        return action

    def update_q_table(self, state, action, reward, next_state):
        state_actions = self.q_table.get(state, {})
        q_predict = state_actions.get(action, 0)

        if next_state in self.q_table:
            q_target = reward + self.gamma * max(self.q_table[next_state].values())
        else:
            q_target = reward

        self.q_table.setdefault(state, {})[action] = q_predict + self.alpha * (q_target - q_predict)

def calculate_reward(actual_price, predicted_price):
    error = abs(actual_price - predicted_price)
    return max(0, 1 - (error / actual_price))

def stock_prediction_with_q_learning(stock_data, lstm_predicted_price, num_episodes=100):
    actions = np.arange(-10, 11)
    agent = QLearningAgent(actions)

    for episode in range(num_episodes):
        for day in range(len(stock_data) - 1):
            state = tuple(stock_data.iloc[day, :].values)
            action = agent.choose_action(state)
            predicted_change = action / 100
            predicted_price = stock_data['Close'].iloc[day] * (1 + predicted_change)
            actual_price = stock_data['Close'].iloc[day + 1]
            reward = calculate_reward(actual_price, predicted_price)
            next_state = tuple(stock_data.iloc[day + 1, :].values)
            agent.update_q_table(state, action, reward, next_state)

    return agent