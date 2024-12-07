import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import torch
from utils.MLP import MLP
from utils.training_utils import EarlyStopping, training_loop

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target  # Already encoded as integers 0, 1, 2

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

    model = MLP(input_dim=X_train_tensor.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    criterion = torch.nn.MSELoss()

    # Train the model
    trained_model = training_loop(
        X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, model, criterion, optimizer, epochs=100, batch_size=32, verbose=True, early_stopping=early_stopping
    )

    # After training, the best model for this fold is saved, and the weights are restored
    print(f"Fold {fold} completed.")
    validation_losses.append(early_stopping.best_loss)  # Track validation loss

# After cross-validation, print the average validation loss
average_loss = np.mean(validation_losses)
print(f"Average validation loss across 5 folds: {average_loss}")

# Optionally, save the model with the lowest validation loss (from the best fold)
best_fold = np.argmin(validation_losses) + 1
print(f"The best fold is fold {best_fold} with validation loss {validation_losses[best_fold-1]}")
