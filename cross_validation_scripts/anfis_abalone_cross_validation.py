import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import torch

import sys
print(sys.path)

from utils.ANFIS import ANFIS
from utils.training_utils import EarlyStopping, training_loop


# Load the dataset
print("Loading dataset from abalone.data...")
data = pd.read_csv("data/abalone.data", header=None)

# Extract features and target
categorical_col = data.iloc[:, 0]  # First column (categorical)
numerical_cols = data.iloc[:, 1:-1]  # All numerical columns except the last one
target_col = data.iloc[:, -1]  # Last column (target)

# One-hot encode the categorical column
encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_categorical = encoder.fit_transform(categorical_col.values.reshape(-1, 1))

# Standardize the numerical columns
scaler = StandardScaler()
scaled_numerical = scaler.fit_transform(numerical_cols.values)

# Combine the scaled numerical and one-hot encoded categorical columns
X = np.hstack([encoded_categorical, scaled_numerical])
y = target_col.values

# Cross-validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold = 0
validation_losses = []

for train_index, val_index in kf.split(X):
    fold += 1
    print(f"Training fold {fold}...")

    # Split the data into train and validation sets
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    # Standardize features for this fold
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=5, delta=0.01, verbose=True, save_best_model=False, best_model_path=f"../models/abalone_best_anfis_model_fold_{fold}.pth")

    # Initialize the ANFIS model
    model = ANFIS(number_of_features=X_train_tensor.shape[1], number_of_membership_functions=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    criterion = torch.nn.MSELoss()

    # Train the model
    trained_model = training_loop(
        X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, model, criterion, optimizer, epochs=100, batch_size=32, verbose=True, early_stopping=early_stopping
    )

    # After training, the best model for this fold is saved, and the weights are restored
    print(f"Fold {fold} completed.")
    validation_losses.append(early_stopping.best_loss)  # Track validation loss

# After cross-validation, calculate statistics and log the results
average_loss = np.mean(validation_losses)
best_loss = np.min(validation_losses)
worst_loss = np.max(validation_losses)

# Format the results
run_losses = " ".join([f"run{fold}: {loss:.4f}" for fold, loss in enumerate(validation_losses, start=1)])
log_message = f"[{run_losses}] avg loss: {average_loss:.4f} best loss: {best_loss:.4f} worst loss: {worst_loss:.4f}"

# Print the log
print(log_message)