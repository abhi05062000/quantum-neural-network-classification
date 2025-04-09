import pennylane as qml
import numpy as np
import torch
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate a toy binary classification dataset
X, y = make_classification(n_samples=100, n_features=2, n_classes=2, random_state=42)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set up quantum device with 2 qubits
dev = qml.device('default.qubit', wires=2)

# Define a simple quantum circuit
@qml.qnode(dev, interface="torch")
def quantum_circuit(weights, x):
    # Encode the data (x) into quantum states using RX gates
    qml.RX(x[0], wires=0)  # Apply RX rotation on qubit 0
    qml.RX(x[1], wires=1)  # Apply RX rotation on qubit 1

    # Apply rotations to the qubits with learnable parameters
    qml.Rot(weights[0], weights[1], weights[2], wires=0)  # Rotation on qubit 0
    qml.Rot(weights[3], weights[4], weights[5], wires=1)  # Rotation on qubit 1

    # Measurement: return the expectation value of Pauli-Z operator on qubit 0
    return qml.expval(qml.PauliZ(0))

# Loss function (mean squared error)
def loss_fn(weights, x, y):
    predictions = quantum_circuit(weights, x)  # Get the predictions from the quantum circuit
    return torch.mean((predictions - y) ** 2)  # Calculate the MSE loss

# Define optimizer
optimizer = torch.optim.Adam([torch.tensor(np.random.randn(6), requires_grad=True)], lr=0.1)

# Training loop
epochs = 100  # Number of training epochs
for epoch in range(epochs):
    epoch_loss = 0
    for i in range(len(X_train)):
        x = torch.tensor(X_train[i], dtype=torch.float32)  # Training input data
        y = torch.tensor(y_train[i], dtype=torch.float32)  # Training labels

        optimizer.zero_grad()  # Reset gradients
        loss = loss_fn(torch.tensor(np.random.randn(6), dtype=torch.float32), x, y)  # Compute loss
        loss.backward()  # Backpropagation (calculate gradients)
        optimizer.step()  # Update parameters (quantum circuit weights)
        
        epoch_loss += loss.item()  # Accumulate loss for reporting

    # Print loss every 10 epochs
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(X_train)}")

# Evaluate the model
predictions = []
for i in range(len(X_test)):
    x = torch.tensor(X_test[i], dtype=torch.float32)  # Test input data
    predictions.append(quantum_circuit(torch.tensor(np.random.randn(6), dtype=torch.float32), x).item())  # Get predictions

# Convert predictions to binary (0 or 1) based on threshold
predictions = [1 if p > 0 else 0 for p in predictions]

# Accuracy
accuracy = accuracy_score(y_test, predictions)  # Compare with true labels
print(f"Accuracy: {accuracy * 100:.2f}%")
