# src/train.py

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import yfinance as yf

# =========================
# 1. DATA LOADING
# =========================
def load_data():
    df = yf.download("AAPL", start="2018-01-01", end="2023-01-01")
    df = df[['Close']]

    # create returns
    df['return'] = df['Close'].pct_change().shift(-1)

    # label: BUY(1), SELL(-1), HOLD(0)
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
    def __init__(self, input_size=1, hidden_size=32):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 3)  # 3 classes

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
            loss = criterion(outputs, y + 1)  # shift labels (-1,0,1 → 0,1,2)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")


# =========================
# 6. MAIN
# =========================
def main():
    print("🚀 Training started...")

    df = load_data()

    X, y = create_sequences(df[['Close']].values, seq_len=10)
    y = df['label'].values[10:]

    dataset = StockDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = LSTMModel()
    train_model(model, loader)

    print("✅ Training completed!")


if __name__ == "__main__":
    main()
