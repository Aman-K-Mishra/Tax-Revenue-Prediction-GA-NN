import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.utils.config import load_config


# ---------- Data prep (similar to load_dataset, but keeps original y) ----------
def prepare_data(csv_path, target_column):
    df = pd.read_csv(csv_path)

    # Handle non-numeric (dates etc.)
    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = pd.to_datetime(df[col], dayfirst=True).dt.year
            except Exception:
                df = df.drop(columns=[col])

    # Keep only numeric
    df = df.select_dtypes(include=[np.number])

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not in dataset")

    X = df.drop(columns=[target_column]).values
    y = df[target_column].values.reshape(-1, 1)

    # Split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    # Scale X and y
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_s = X_scaler.fit_transform(X_train)
    X_val_s   = X_scaler.transform(X_val)
    X_test_s  = X_scaler.transform(X_test)

    y_train_s = y_scaler.fit_transform(y_train)
    y_val_s   = y_scaler.transform(y_val)
    y_test_s  = y_scaler.transform(y_test)

    return (
        X_train_s, X_val_s, X_test_s,
        y_train_s, y_val_s, y_test_s,
        y_train, y_val, y_test,
        X_scaler, y_scaler
    )


# ---------- Model builder ----------
def build_model(input_dim, layers, activation):
    act_layer = nn.ReLU if activation == "relu" else nn.Tanh

    modules = []
    last_dim = input_dim

    for h in layers:
        modules.append(nn.Linear(last_dim, h))
        modules.append(act_layer())
        last_dim = h

    modules.append(nn.Linear(last_dim, 1))
    return nn.Sequential(*modules)


# ---------- Training helper ----------
def train_model(model, X, y, epochs=300, lr=0.001):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    X = torch.tensor(X, dtype=torch.float32).to(device)
    y = torch.tensor(y, dtype=torch.float32).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

    return model


# ---------- Prediction helper ----------
def predict(model, X):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    X = torch.tensor(X, dtype=torch.float32).to(device)

    with torch.no_grad():
        pred = model(X).cpu().numpy()
    return pred


def main():
    cfg = load_config()

    print("\n=== Retraining Best Architecture (300 epochs) ===")

    (
        X_train_s, X_val_s, X_test_s,
        y_train_s, y_val_s, y_test_s,
        y_train_orig, y_val_orig, y_test_orig,
        X_scaler, y_scaler
    ) = prepare_data(cfg.csv_path, cfg.target_column)

    # Load best architecture from GA
    arch_path = "models/best_architecture.json"
    if not os.path.exists(arch_path):
        print("ERROR: models/best_architecture.json not found. Run GA search first.")
        return

    with open(arch_path, "r") as f:
        best_arch = json.load(f)

    print("\nBest architecture:", best_arch)

    input_dim = X_train_s.shape[1]

    # -------------------------------------------------
    # 1) Train on TRAIN+VAL → Evaluate on TEST
    # -------------------------------------------------
    print("\n--- Phase 1: Train on train+val, evaluate on test ---")

    X_trainval_s = np.vstack([X_train_s, X_val_s])
    y_trainval_s = np.vstack([y_train_s, y_val_s])

    model_eval = build_model(
        input_dim=input_dim,
        layers=best_arch["layers"],
        activation=best_arch["activation"],
    )

    model_eval = train_model(
        model_eval,
        X_trainval_s,
        y_trainval_s,
        epochs=300,
        lr=cfg.learning_rate if hasattr(cfg, "learning_rate") else 0.001,
    )

    # Predict on test (scaled)
    y_pred_test_s = predict(model_eval, X_test_s)

    # Inverse transform to original scale
    y_pred_test = y_scaler.inverse_transform(y_pred_test_s)
    y_test_true = y_test_orig  # already original

    # Compute metrics
    mse = mean_squared_error(y_test_true, y_pred_test)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_test_true, y_pred_test)
    r2 = r2_score(y_test_true, y_pred_test)

    print("\n=== Test Set Evaluation (original scale) ===")
    print(f"mse:  {mse}")
    print(f"rmse: {rmse}")
    print(f"mae:  {mae}")
    print(f"r2:   {r2}")

    os.makedirs("models", exist_ok=True)
    torch.save(model_eval.state_dict(), "models/model_test_eval.pth")

    # -------------------------------------------------
    # 2) Train FINAL model on ALL DATA (deploy model)
    # -------------------------------------------------
    print("\n--- Phase 2: Train on ALL data (train+val+test) for deployment ---")

    X_all_s = np.vstack([X_train_s, X_val_s, X_test_s])
    y_all_s = np.vstack([y_train_s, y_val_s, y_test_s])

    model_final = build_model(
        input_dim=input_dim,
        layers=best_arch["layers"],
        activation=best_arch["activation"],
    )

    model_final = train_model(
        model_final,
        X_all_s,
        y_all_s,
        epochs=300,
        lr=cfg.learning_rate if hasattr(cfg, "learning_rate") else 0.001,
    )

    # Save final model + scalers
    torch.save(model_final.state_dict(), "models/final_model.pth")

    with open("models/final_X_scaler.pkl", "wb") as f:
        pickle.dump(X_scaler, f)

    with open("models/final_y_scaler.pkl", "wb") as f:
        pickle.dump(y_scaler, f)

    print("\n=== Final model saved ===")
    print("Model:   models/final_model.pth")
    print("X scaler: models/final_X_scaler.pkl")
    print("y scaler: models/final_y_scaler.pkl")


if __name__ == "__main__":
    main()
