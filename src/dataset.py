import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import random_split
from sklearn.utils import class_weight

def preprocess_data(data):
    for column in data.columns:
        if data[column].dtype == object:
            le = LabelEncoder()
            data[column] = le.fit_transform(data[column])

    # Drop unnecessary columns
    data = data.drop(columns=['Flow ID', 'Src IP', 'Dst IP', 'Timestamp', 'Label.1'])
    
    # Replace infinity with NaN
    data.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Replace NaN with the average value of the respective column
    #data.fillna(data.mean(), inplace=True)
    # Remove rows containing NaN values
    data = data.dropna()
    
    # Perform min-max normalization on feature columns
    feature_columns = data.columns[data.columns != 'Label']  # Exclude Label column
    scaler = MinMaxScaler()
    data[feature_columns] = scaler.fit_transform(data[feature_columns])

    # Split features and labels
    X = data.drop(columns=["Label"]).values
    y = data["Label"].values

    return X, y

def load_data(train_file_path):
    # Load training data
    data = pd.read_csv(train_file_path)
    X, y = preprocess_data(data)
    return X, y

def flip_labels(y, flip_label, new_label, flip_percentage):
    np.random.seed(42)
    
    # Find indices of instances with the flip label
    flip_indices = np.where(y == flip_label)[0]
    print(flip_indices)
    
    # Calculate number of flips based on percentage
    num_flips = int(len(flip_indices) * flip_percentage / 100)
    
    # Select random indices to flip without shuffling
    flip_indices_to_flip = np.random.choice(flip_indices, size=num_flips, replace=False)
    print(flip_indices_to_flip)
    
    # Update labels at selected indices
    y[flip_indices_to_flip] = new_label
    
    return y
    

def prepare_dataset_attack(DATASET_PATH: str, num_partitions: int, batch_size: int, malicious_clients: list, flip_label: int, new_label: int, flip_percentage: float):
    # Set random seed for reproducibility
    np.random.seed(42)
    # Load data
    file_path = DATASET_PATH
    X_train, y_train = load_data(file_path)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=30, stratify=y_train)

    # Calculate class counts
    class_counts = {label: np.sum(y_train == label) for label in np.unique(y_train)}
    
    # Calculate the size of each partition based on class proportions
    partition_sizes = {label: count // num_partitions for label, count in class_counts.items()}
    remainder = {label: count % num_partitions for label, count in class_counts.items()}
    
    partitions_X = [[] for _ in range(num_partitions)]
    partitions_y = [[] for _ in range(num_partitions)]
    
    current_partition_index = {label: 0 for label in class_counts.keys()}
    
    # Assign data points to partitions while maintaining class proportions
    for X, y in zip(X_train, y_train):
        label = y
        partition_index = current_partition_index[label]
        partitions_X[partition_index].append(X)
        partitions_y[partition_index].append(y)
        current_partition_index[label] = (current_partition_index[label] + 1) % num_partitions

    
    # Convert lists to numpy arrays
    partitions_X = [np.array(partition_X) for partition_X in partitions_X]
    partitions_y = [np.array(partition_y) for partition_y in partitions_y]
    
    print("-----------DATASET INFORMATION ---------------")

    # Print class counts of each partition before flipping
    print("Class counts of each partition before flipping:")
    for i, y_part in enumerate(partitions_y):
        print(partitions_y[i][:100])
        class_count = {label: np.sum(y_part == label) for label in np.unique(y_train)}
        print(f"Partition {i+1}: {class_count}")
    
    # Flip labels in the specified partition
    for i in range(len(partitions_y)):
        if i in malicious_clients:
            partitions_y[i] = flip_labels(partitions_y[i], flip_label, new_label, flip_percentage)
    
    # Print class counts of each partition after flipping
    print("Class counts of each partition after flipping:")
    for i, y_part in enumerate(partitions_y):
        print(partitions_y[i][:100])
        class_count = {label: np.sum(y_part == label) for label in np.unique(y_train)}
        print(f"Partition {i+1}: {class_count}")
    
    # Create DataLoader for each partition
    train_loaders = []
    for X_part, y_part in zip(partitions_X, partitions_y):
        train_set = TensorDataset(torch.tensor(X_part, dtype=torch.float32), torch.tensor(y_part, dtype=torch.long))
        train_loader = DataLoader(train_set, batch_size=batch_size,shuffle=False)
        train_loaders.append(train_loader)
    
    # Create DataLoader for the testing set
    test_set = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    test_loader = DataLoader(test_set, batch_size=batch_size,shuffle=False)
    # Print values in the first batch of each train loader
    print("Values in the first batch of train loaders:")
    for i, train_loader in enumerate(train_loaders):
        print(f"Partition {i+1}:")
        data, target = next(iter(train_loader))
        print("Target:")
        print(target)
    
    return train_loaders, test_loader

def prepare_dataset_no_attack(DATASET_PATH: str, num_partitions: int, batch_size: int):
    # Set random seed for reproducibility
    np.random.seed(42)
    # Load data
    file_path = DATASET_PATH
    X_train, y_train = load_data(file_path)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=30, stratify=y_train)

    # Calculate class counts
    class_counts = {label: np.sum(y_train == label) for label in np.unique(y_train)}
    
    # Calculate the size of each partition based on class proportions
    partition_sizes = {label: count // num_partitions for label, count in class_counts.items()}
    remainder = {label: count % num_partitions for label, count in class_counts.items()}
    
    partitions_X = [[] for _ in range(num_partitions)]
    partitions_y = [[] for _ in range(num_partitions)]
    
    current_partition_index = {label: 0 for label in class_counts.keys()}
    
    # Assign data points to partitions while maintaining class proportions
    for X, y in zip(X_train, y_train):
        label = y
        partition_index = current_partition_index[label]
        partitions_X[partition_index].append(X)
        partitions_y[partition_index].append(y)
        current_partition_index[label] = (current_partition_index[label] + 1) % num_partitions

    
    # Convert lists to numpy arrays
    partitions_X = [np.array(partition_X) for partition_X in partitions_X]
    partitions_y = [np.array(partition_y) for partition_y in partitions_y]
    
    print("-----------DATASET INFORMATION ---------------")

    # Print class counts of each partition 
    print("Class counts of each partition:")
    for i, y_part in enumerate(partitions_y):
        print(partitions_y[i][:100])
        class_count = {label: np.sum(y_part == label) for label in np.unique(y_train)}
        print(f"Partition {i+1}: {class_count}")
    
    # Create DataLoader for each partition
    train_loaders = []
    for X_part, y_part in zip(partitions_X, partitions_y):
        train_set = TensorDataset(torch.tensor(X_part, dtype=torch.float32), torch.tensor(y_part, dtype=torch.long))
        train_loader = DataLoader(train_set, batch_size=batch_size,shuffle=False)
        train_loaders.append(train_loader)
    
    # Create DataLoader for the testing set
    test_set = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    test_loader = DataLoader(test_set, batch_size=batch_size,shuffle=False)
    # Print values in the first batch of each train loader
    print("Values in the first batch of train loaders:")
    for i, train_loader in enumerate(train_loaders):
        print(f"Partition {i+1}:")
        data, target = next(iter(train_loader))
        print("Target:")
        print(target)
    
    return train_loaders, test_loader

def prepare_dataset_validators(DATASET_PATH: str, num_partitions: int, batch_size: int):
    # Set random seed for reproducibility
    np.random.seed(42)
    # Load data
    file_path = DATASET_PATH
    X_train, y_train = load_data(file_path)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=30, stratify=y_train)

    # Calculate class counts
    class_counts = {label: np.sum(y_train == label) for label in np.unique(y_train)}
    
    # Calculate the size of each partition based on class proportions
    partition_sizes = {label: count // num_partitions for label, count in class_counts.items()}
    remainder = {label: count % num_partitions for label, count in class_counts.items()}
    
    partitions_X = [[] for _ in range(num_partitions)]
    partitions_y = [[] for _ in range(num_partitions)]
    
    current_partition_index = {label: 0 for label in class_counts.keys()}
    
    # Assign data points to partitions while maintaining class proportions
    for X, y in zip(X_train, y_train):
        label = y
        partition_index = current_partition_index[label]
        partitions_X[partition_index].append(X)
        partitions_y[partition_index].append(y)
        current_partition_index[label] = (current_partition_index[label] + 1) % num_partitions

    
    # Convert lists to numpy arrays
    partitions_X = [np.array(partition_X) for partition_X in partitions_X]
    partitions_y = [np.array(partition_y) for partition_y in partitions_y]
    
    print("-----------DATASET INFORMATION ---------------")

    # Print class counts of each partition 
    print("Class counts of each partition:")
    for i, y_part in enumerate(partitions_y):
        print(partitions_y[i][:100])
        class_count = {label: np.sum(y_part == label) for label in np.unique(y_train)}
        print(f"Partition {i+1}: {class_count}")
    
    # Create DataLoader for each partition
    train_loaders = []
    for X_part, y_part in zip(partitions_X, partitions_y):
        train_set = TensorDataset(torch.tensor(X_part, dtype=torch.float32), torch.tensor(y_part, dtype=torch.long))
        train_loader = DataLoader(train_set, batch_size=batch_size,shuffle=False)
        train_loaders.append(train_loader)
    
    # Create DataLoader for the testing set
    test_set = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    test_loader = DataLoader(test_set, batch_size=batch_size,shuffle=False)
    # Print values in the first batch of each train loader
    print("Values in the first batch of train loaders:")
    for i, train_loader in enumerate(train_loaders):
        print(f"Partition {i+1}:")
        data, target = next(iter(train_loader))
        print("Target:")
        print(target)
    
    return train_loaders, test_loader


