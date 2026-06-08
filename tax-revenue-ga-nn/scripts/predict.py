import torch
import torch.nn as nn
import numpy as np
import pickle
import json
import os

from src.utils.config import load_config


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


def main():
    cfg = load_config()

    # -------- Load scalers --------
    with open("models/final_X_scaler.pkl", "rb") as f:
        X_scaler = pickle.load(f)

    with open("models/final_y_scaler.pkl", "rb") as f:
        y_scaler = pickle.load(f)

    # -------- Load best architecture --------
    with open("models/best_architecture.json", "r") as f:
        arch = json.load(f)

    layers = arch["layers"]
    activation = arch["activation"]

    input_dim = len(cfg.predict_features)

    # -------- Build model --------
    model = build_model(
        input_dim=input_dim,
        layers=layers,
        activation=activation
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.load_state_dict(torch.load("models/final_model.pth", map_location=device))
    model = model.to(device)
    model.eval()

    # -------- User Inputs --------

    print("\n=== Tax Revenue Prediction ===")

    user_vals = []
    for feat in cfg.predict_features:
        val = float(input(f"Enter {feat}: "))
        user_vals.append(val)

    X_input = np.array(user_vals).reshape(1, -1)

    # Scale input
    X_scaled = X_scaler.transform(X_input)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

    # Predict
    with torch.no_grad():
        pred_scaled = model(X_tensor).cpu().numpy()

    pred_final = y_scaler.inverse_transform(pred_scaled)[0][0]

    print("\nPredicted Tax Revenue:", round(pred_final, 2))


if __name__ == "__main__":
    main()
