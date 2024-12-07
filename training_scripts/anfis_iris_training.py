from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import torch

from utils.ANFIS import ANFIS
from utils.training_utils import EarlyStopping, training_loop, train_valid_test_split

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target  # Already encoded as integers 0, 1, 2

# Split the dataset
x_train, y_train, x_val, y_val, x_test, y_test = train_valid_test_split(X, y)

# Standardize features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

# Combine features and target for the test set
test_data = pd.DataFrame(x_test)  # Create DataFrame for features
test_data['Target'] = y_test      # Add the target column

# Save to CSV
test_data.to_csv('test_datasets/iris_test_dataset.csv', index=False, header=False)
print("Test dataset saved to test_datasets/iris_test_dataset.csv")

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)  # Single-output regression
X_val_tensor = torch.tensor(x_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

# Define the ANFIS model
model = ANFIS(
    number_of_features=X_train_tensor.shape[1],
    number_of_membership_functions=3
)

# Define optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()  # Single-output regression

# Early stopping
early_stopping = EarlyStopping(patience=5, delta=0.01, verbose=True, save_best_model=True, best_model_path="models/iris_best_anfis_model.pth")

# Train the model
trained_model = training_loop(
    X_train_tensor, y_train_tensor, 
    X_val_tensor, y_val_tensor,
    model, criterion, optimizer, 
    epochs=100, batch_size=16, 
    verbose=True, early_stopping=early_stopping
)

# After training, the best model is saved and the weights are restored to the model.