from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

def load_data(input_data, target_data_sst3, target_data_sst8, device, batch_size=16):
    # Split input and target data into training and test sets for SST3 and SST8
    X_train, X_test, y_train_sst3, y_test_sst3 = train_test_split(input_data, target_data_sst3, test_size=0.25, random_state=0)
    _, _, y_train_sst8, y_test_sst8 = train_test_split(input_data, target_data_sst8, test_size=0.25, random_state=0)

    # Move data to the device (GPU if available) for training
    X_train, X_test = X_train.clone().detach().to(device), X_test.clone().detach().to(device)
    y_train_sst3, y_test_sst3 = y_train_sst3.clone().detach().to(device), y_test_sst3.clone().detach().to(device)
    y_train_sst8, y_test_sst8= y_train_sst8.clone().detach().to(device), y_test_sst8.clone().detach().to(device)

    # Create datasets and loaders for training and testing
    train_dataset = TensorDataset(X_train, y_train_sst3, y_train_sst8)
    test_dataset = TensorDataset(X_test, y_test_sst3, y_test_sst8)

    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader