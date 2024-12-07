import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
import torch
from utils.ANFIS import ANFIS
from utils.training_utils import train_valid_test_split, EarlyStopping, training_loop

# Load the dataset from Excel
print("Loading dataset from powerPlant.csv...")
data = pd.read_csv("data/powerPlant.csv")

# Split features and target
X = data.iloc[:, :-1].values  # All columns except the last one as features
y = data.iloc[:, -1].values   # The last column as the target

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
test_data.to_csv('test_datasets/power_plant_test_dataset.csv', index=False, header=False)
print("Test dataset saved to test_datasets/power_plant_test_dataset.csv")


# Convert to PyTorch tensors
X_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_val_tensor = torch.tensor(x_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

early_stopping = EarlyStopping(patience=5, delta=0.01, verbose=True, save_best_model=True, best_model_path="models/power_plant_best_anfis_model.pth")

model = ANFIS(
    number_of_features=X_train_tensor.shape[1],
    number_of_membership_functions=3
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
criterion = torch.nn.MSELoss()

trained_model = training_loop(
    x_train, y_train, x_val, y_val, model, criterion, optimizer, epochs=100, batch_size=32, verbose=True, early_stopping=early_stopping
)

# After training, the best model is saved and the weights are restored to the model.

