import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import torch
from utils.MLP import MLP
from utils.training_utils import train_valid_test_split, EarlyStopping, training_loop

# Load the dataset
print("Loading dataset from abalone.data...")
data = pd.read_csv("data/abalone.data", header=None)  # Ensure no header if using raw data

# Extract features and target
categorical_col = data.iloc[:, 0]  # First column (categorical)
numerical_cols = data.iloc[:, 1:-1]  # All numerical columns except the last one
target_col = data.iloc[:, -1]  # Last column (target)

# One-hot encode the categorical column
encoder = OneHotEncoder(sparse_output=False, drop='first')  # Drop one category to avoid multicollinearity
encoded_categorical = encoder.fit_transform(categorical_col.values.reshape(-1, 1))  # Ensure 2D shape

# Standardize the numerical columns
scaler = StandardScaler()
scaled_numerical = scaler.fit_transform(numerical_cols.values)

# Combine the scaled numerical and one-hot encoded categorical columns
X = np.hstack([encoded_categorical, scaled_numerical])
# Target column as a separate array
y = target_col.values

x_train, y_train, x_val, y_val, x_test, y_test = train_valid_test_split(X, y)

# Standardize features
scaler = StandardScaler()
x_train= scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

# Combine features and target for the test set
test_data = pd.DataFrame(x_test)  # Create DataFrame for features
test_data['Target'] = y_test      # Add the target column

# Save to CSV
test_data.to_csv('test_datasets/abalone_test_dataset.csv', index=False, header=False)
print("Test dataset saved to test_datasets/abalone_test_dataset.csv")


# Convert to PyTorch tensors
X_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_val_tensor = torch.tensor(x_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

early_stopping = EarlyStopping(patience=5, delta=0.01, verbose=True, save_best_model=True, best_model_path="models/abalone_best_mlp_model.pth")

model = MLP(input_dim=X_train_tensor.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
criterion = torch.nn.MSELoss()

trained_model = training_loop(
    x_train, y_train, x_val, y_val, model, criterion, optimizer, epochs=100, batch_size=32, verbose=True, early_stopping=early_stopping
)

# After training, the best model is saved and the weights are restored to the model.
