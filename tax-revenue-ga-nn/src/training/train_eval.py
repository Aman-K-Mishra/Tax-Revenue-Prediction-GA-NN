import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def train_and_eval(chrom, X_train, y_train, X_val, y_val, epochs=200, lr=0.001):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build model
    layers = []
    input_dim = X_train.shape[1]
    last_dim = input_dim

    for h in chrom["layers"]:
        layers.append(nn.Linear(last_dim, h))
        layers.append(nn.ReLU() if chrom["activation"] == "relu" else nn.Tanh())
        last_dim = h

    layers.append(nn.Linear(last_dim, 1))
    model = nn.Sequential(*layers).to(device)

    # Data
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.float32).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train)
        loss = criterion(pred, y_train)
        loss.backward()
        optimizer.step()

    # Validation loss
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val)
        val_loss = criterion(val_pred, y_val).item()

    return val_loss
