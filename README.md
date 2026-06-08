Tax Revenue Prediction using GA-Optimized Neural Network
A machine learning project that uses a Genetic Algorithm (GA) to search for an effective neural network architecture for tax revenue prediction from macroeconomic indicators. The best model is then retrained and used for inference through a simple Python prediction interface.
Overview
This project predicts yearly tax revenue using features such as:
GDP
inflation
population
imports
exports
corporate tax rate
Because the dataset is small and the relationship between the variables is nonlinear, a genetic algorithm is used to automatically explore neural network architectures instead of relying on manual hyperparameter tuning.
Features
Genetic Algorithm for neural architecture search
Standardized preprocessing with feature scalers
Train/validation/test pipeline
Model retraining for the best architecture
Saved PyTorch model and scalers for inference
Interactive CLI prediction script
Lightweight design suited for small structured datasets
How It Works
The input data is scaled using `StandardScaler`.
A GA evolves candidate neural network architectures.
Each architecture is trained briefly and evaluated on validation MSE.
The best-performing architecture is retrained more thoroughly.
The final model and scalers are saved for future predictions.
A CLI script loads the saved artifacts and produces tax revenue forecasts.
Model Summary
The final selected architecture is a compact feedforward neural network with two hidden layers.
Reported final performance:
R²: 0.827
RMSE: 69.5k
MAE: 55.4k
MSE: 4.8 billion
These results suggest that the model captures the relationship between macroeconomic variables and tax revenue well enough for practical forecasting use.
Repository Structure
```bash
.
├── data/
├── models/
│   ├── final_model.pth
│   ├── final_X_scaler.pkl
│   └── final_y_scaler.pkl
├── predict.py
├── train.py
├── ga_search.py
├── requirements.txt
└── README.md
```
> File names may vary depending on how you organized the repo.
Installation
Clone the repository:
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```
Create and activate a virtual environment:
```bash
python -m venv venv
```
On Windows:
```bash
venv\Scripts\activate
```
On macOS/Linux:
```bash
source venv/bin/activate
```
Install dependencies:
```bash
pip install -r requirements.txt
```
Usage
1. Train the model
```bash
python train.py
```
This will:
load and preprocess the dataset,
run GA-based architecture search,
retrain the best network,
save the trained model and scalers.
2. Make a prediction
```bash
python predict.py
```
Then enter the macroeconomic values when prompted.
Example input:
```text
GDP: 1234567
Inflation: 5.2
Population: 1400000000
Imports: 340000
Exports: 290000
Corporate Tax Rate: 25
```
The script will output the predicted tax revenue.
Data
The project uses a yearly macroeconomic dataset with 129 observations.  
The dataset is small, clean, and suitable for experimenting with evolutionary optimization methods on structured tabular data.
Why Genetic Algorithms?
A genetic algorithm is useful here because it can automatically explore different neural network designs and reduce manual tuning. This is especially helpful when:
the dataset is small,
the feature relationships are nonlinear,
and you want a compact model instead of an overly deep network.
Future Improvements
Replace synthetic or structured data with real government economic data
Add cross-validation for stronger evaluation
Try LSTM or Transformer-based forecasting
Build a web dashboard for forecasting
Add batch prediction support from CSV input
Tech Stack
Python
PyTorch
NumPy
Pandas
scikit-learn
Genetic Algorithm logic in Python
License
Add your preferred license here.
Acknowledgments
Built as a machine learning project for tax revenue forecasting using genetic algorithms and neural networks.
