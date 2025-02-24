import torch

# Custom accuracy calculation function for Q3 accuracy
def q3_acc(y_true, y_pred):
    y = y_true
    y_ = torch.argmax(y_pred, dim=1)  #[batch_size, seq_len]
    mask = y > 0                      # Mask to ignore padded parts of sequences
    y = torch.masked_select(y, mask)
    y_ = torch.masked_select(y_, mask)
    return torch.mean((y == y_).float())    # Calculate mean accuracy

# Define loss functions for SST3 and SST8 predictions
loss_sst3 = torch.nn.CrossEntropyLoss()
loss_sst8 = torch.nn.CrossEntropyLoss()
