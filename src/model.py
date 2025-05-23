import torch
import torch.nn as nn
import torch.nn.functional as F

# Note the model and functions here defined do not have any FL-specific components.
# Set random seed for reproducibility
torch.manual_seed(42)

# Define the neural network model with the input size
input_size = 79
num_classes=4
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)  # First hidden layer with 150 neurons
        self.fc2 = nn.Linear(64, 32)         # Second hidden layer with 100 neurons
        self.fc3 = nn.Linear(32, num_classes)          # Third hidden layer with 50 neurons

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x # Apply softmax activation function along the class dimension