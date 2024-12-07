from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from utils.ANFIS import ANFIS
import torch
import pandas as pd

class EarlyStopping:
    def __init__(self, patience=5, delta=0.0, verbose=True, save_best_model=False, best_model_path="best_model.pth"):
        """
        Args:
            patience (int): Number of epochs to wait after last best validation loss.
            delta (float): Minimum change to qualify as an improvement.
            verbose (bool): Print message when stopping early.
            save_best_model (bool): Whether to save the model with the best validation loss.
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.save_best_model = save_best_model
        self.best_loss = None
        self.counter = 0
        self.early_stop = False
        self.best_model_wts = None
        self.best_model_path = best_model_path

    def __call__(self, val_loss, model, epoch):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_wts = model.state_dict()
            if self.verbose:
                print(f"Epoch {epoch + 1}: EarlyStopping initialized with best_loss={val_loss:.4f}")
        elif val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.best_model_wts = model.state_dict()
            self.counter = 0
            if self.verbose:
                print(f"Epoch {epoch + 1}: Validation loss improved to {val_loss:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"Epoch {epoch + 1}: No improvement. EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.save_best_model:
                    print(f"Saving the best model with loss={self.best_loss:.4f}")
                    torch.save(self.best_model_wts, self.best_model_path)

        return self.early_stop

def train_valid_test_split(X, y, train_ratio=0.75, validation_ratio=0.15, test_ratio=0.10):
    # train is now 75% of the entire data set
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=1 - train_ratio, random_state=42)

    # test is now 10% of the initial data set
    # validation is now 15% of the initial data set
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio), shuffle=False) 

    return x_train, y_train, x_val, y_val, x_test, y_test

def training_loop(
    x_train, y_train, x_val, y_val, model, criterion, optimizer, epochs, batch_size=32, verbose=True, early_stopping=None
):
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    x_val_tensor = torch.tensor(x_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

    # Training loop 
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(X_train_tensor.size(0))
        epoch_loss = 0

        for i in range(0, X_train_tensor.size(0), batch_size):
            indices = permutation[i:i + batch_size]
            batch_X, batch_y = X_train_tensor[indices], y_train_tensor[indices]

            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_X.size(0)  # Scale by batch size

        # Calculate average training loss for the epoch
        avg_epoch_loss = epoch_loss / len(X_train_tensor)
        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_epoch_loss:.4f}")

        # Validation loss for early stopping
        model.eval()
        with torch.no_grad():
            val_predictions = model(x_val_tensor).detach()
            val_loss = criterion(val_predictions, y_val_tensor).item() 

        # Print epoch summary if verbose
        if verbose:
            print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_epoch_loss:.4f}, Validation Loss: {val_loss:.4f}")

        # Early stopping check
        if early_stopping(val_loss, model, epoch):
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break


    return model

