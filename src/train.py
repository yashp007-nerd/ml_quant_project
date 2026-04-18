# src/train.py

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import yfinance as yf
import matplotlib.pyplot as plt
import os


# =========================
# 1. LOAD DATA + FEATURES
# =========================
def load_data():
    df = yf.download("AAPL", start="2018-01-01", end="2023-01-01")

    df['return'] = df['Close'].pct_change().shift(-1)

    # Moving Average
    df['ma'] = df['Close'].rolling(10).mean()

    # EMA
    df['ema'] = df['Close'].ewm(span=10).mean()

    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # Label
    def label_fn(x):
        if x > 0.002:
            return 1
        elif x < -0.002:
            return -1
        else:
            return 0

    df['label'] = df['return'].apply(label_fn)
    df = df.dropna()

    return df


# =========================
# 2. CREATE SEQUENCES
# =========================
def create_sequences(data, seq_len=10):
    X, y = [], []

    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])

    return np.array(X), np.array(y)


# =========================
# 3. DATASET
# =========================
class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =========================
# 4. MODEL
# =========================
class LSTMModel(nn.Module):
    def __init__(self, input_size=4, hidden_size=32):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 3)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


# =========================
# 5. TRAINING
# =========================
def train_model(model, loader, epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0

        for X, y in loader:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(X)

            loss = criterion(outputs, y + 1)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")


# =========================
# 6. BACKTEST
# =========================
def backtest(model, dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    capital = 10000
    position = 0
    cost = 0.001

    prices = dataset.X[:, -1, 0].numpy()
    equity_curve = []

    with torch.no_grad():
        for i in range(len(dataset)):
            X, _ = dataset[i]
            X = X.unsqueeze(0).to(device)

            output = model(X)
            pred = torch.argmax(output).item() - 1

            price = prices[i]

            if pred == 1 and position == 0:
                position = (capital * (1 - cost)) / price
                capital = 0

            elif pred == -1 and position > 0:
                capital = position * price * (1 - cost)
                position = 0

            total_value = capital + position * price
            equity_curve.append(total_value)

    return equity_curve


# =========================
# 7. BUY & HOLD
# =========================
def buy_and_hold(prices):
    initial = 10000
    shares = initial / prices[0]
    return shares * prices


# =========================
# 8. METRICS
# =========================
def compute_metrics(equity_curve):
    returns = np.diff(equity_curve) / equity_curve[:-1]

    sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)

    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak
    max_dd = drawdown.min()

    return sharpe, max_dd


# =========================
# 9. MAIN
# =========================
def main():
    print("🚀 Training started...")

    df = load_data()

    features = df[['Close', 'ma', 'ema', 'rsi']].values

    X, _ = create_sequences(features, seq_len=10)
    y = df['label'].values[10:]

    # Train/Test split
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    train_dataset = StockDataset(X_train, y_train)
    test_dataset = StockDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = LSTMModel()
    train_model(model, train_loader)

    print("📊 Backtesting on TEST data...")
    equity_curve = backtest(model, test_dataset)

    prices = test_dataset.X[:, -1, 0].numpy()
    bh_curve = buy_and_hold(prices)

    sharpe, max_dd = compute_metrics(equity_curve)

    print(f"💰 Strategy Final Value: {equity_curve[-1]:.2f}")
    print(f"📈 Buy & Hold Value: {bh_curve[-1]:.2f}")
    print(f"📊 Sharpe Ratio: {sharpe:.2f}")
    print(f"📉 Max Drawdown: {max_dd:.2%}")

    # 🔥 CREATE RESULTS FOLDER
    os.makedirs("results", exist_ok=True)

    # 🔥 PLOT + SAVE
    plt.figure(figsize=(12,6))
    plt.plot(equity_curve, label="Strategy", linewidth=2)
    plt.plot(bh_curve, label="Buy & Hold", linewidth=2)
    plt.title("Strategy vs Buy & Hold (Out-of-Sample)", fontsize=14)
    plt.xlabel("Time")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid(True)

    plt.savefig("results/equity_curve.png")  # SAVE
    plt.close()

    print("📸 Plot saved at: results/equity_curve.png")
    print("✅ Done!")


if __name__ == "__main__":
    main()
