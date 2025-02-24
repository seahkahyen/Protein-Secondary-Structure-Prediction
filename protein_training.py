import pandas as pd
import torch

import data.vectorisation as vectorisation 
import data.data_setup as data_setup
import model.transformer as transformer

from model.training_utils import q3_acc, loss_sst3, loss_sst8
from evaluation.plots import plot_training_history

# Load the cleaned protein sequence dataset
df = pd.read_csv('2018-06-06-ss.cleaned.csv')
maxlen_seq = 192
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the tokenised preprocessed input data
input_data, target_data_sst3, target_data_sst8, n_words, \
    n_tags_sst3, n_tags_sst8, _, _, _, _, _ = vectorisation.tokenisation(df, maxlen_seq, device)

# Display shapes of processed data for verification
print(f"Input shape: {input_data.shape}")
print(f"sst3 Target shape: {target_data_sst3.shape}")
print(f"sst8 Target shape: {target_data_sst8.shape}")

# Define hyperparameters        
embed_dim = 512             # Embedding dimension for token embeddings
num_heads = 16              # Number of attention heads in transformer
ff_dim = 2048               # Dimension of feedforward network within transformer
dropout = 0.2               # Dropout rate for regularization
num_encoder_layers=6        # Number of stacked transformer encoder layers
batch_size = 16

# Initialize model
model = transformer.TransformerModel(n_words, n_tags_sst3, n_tags_sst8, embed_dim, 
                                     num_heads, ff_dim, maxlen_seq, num_encoder_layers, dropout)
model = model.to(device)

train_loader, test_loader = data_setup.load_data(input_data, target_data_sst3, target_data_sst8, device, batch_size=16)

# Set up optimizer (Adam) for model parameter updates
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Initialize lists to store history of training and validation metrics
train_loss_sst3_history = []
train_loss_sst8_history = []
train_q3_acc_history = []
train_q8_acc_history = []

val_loss_sst3_history = []
val_loss_sst8_history = []
val_q3_acc_history = []
val_q8_acc_history = []

# Define training loop parameters
epochs = 40
plot_epoch = epochs
patience = 3 
best_val_loss = float('inf')
epochs_no_improve = 0

# Training loop over the specified number of epochs
for epoch in range(epochs):
    model.train()
    total_loss_sst3 = 0
    total_loss_sst8 = 0
    total_q3_acc = 0.0
    total_q8_acc = 0.0

    # Iterate over each batch in the training data
    for inputs, targets_sst3, targets_sst8 in train_loader:
        inputs, targets_sst3, targets_sst8 = inputs.to(device), targets_sst3.to(device), targets_sst8.to(device)
        
        outputs_sst3, outputs_sst8 = model(inputs)
        
        # Adjust output dimensions to [batch_size, num_classes, seq_len] for loss calculation
        outputs_sst3 = outputs_sst3.permute(1, 2, 0)  #[batch_size, num_classes, seq_len]
        outputs_sst8 = outputs_sst8.permute(1, 2, 0)  #[batch_size, num_classes, seq_len]
        
        # print(outputs_sst3.shape)
        # print(targets_sst3.shape)

        # Calculate loss for SST3 and SST8 predictions
        loss_sst3_value = loss_sst3(outputs_sst3, targets_sst3)  
        loss_sst8_value = loss_sst8(outputs_sst8, targets_sst8)
        loss = loss_sst3_value + loss_sst8_value
        optimizer.zero_grad()
        
        # Backpropagation and optimizer step
        loss.backward()
        optimizer.step()

        # Track losses and accuracies for each batch
        total_loss_sst3 += loss_sst3_value.item()
        total_loss_sst8 += loss_sst8_value.item()
        
        batch_q3_acc = q3_acc(targets_sst3, outputs_sst3)
        batch_q8_acc = q3_acc(targets_sst8, outputs_sst8)

        total_q3_acc += batch_q3_acc.item()
        total_q8_acc += batch_q8_acc.item()
    
    # Calculate average loss and accuracy for the epoch
    avg_loss_sst3 = total_loss_sst3 / len(train_loader)
    avg_loss_sst8 = total_loss_sst8 / len(train_loader)
    avg_q3_training_acc = total_q3_acc / len(train_loader)
    avg_q8_training_acc = total_q8_acc / len(train_loader)

    # Store metrics history for plotting later
    train_loss_sst3_history.append(avg_loss_sst3)
    train_loss_sst8_history.append(avg_loss_sst8)
    train_q3_acc_history.append(avg_q3_training_acc)
    train_q8_acc_history.append(avg_q8_training_acc)

    print(f"Epoch [{epoch+1}/{epochs}] \nTraining Loss SST3: {avg_loss_sst3:.4f}, Training Loss SST8: {avg_loss_sst8:.4f}")
    print(f"Training Q3 Accuracy: {avg_q3_training_acc:.4f}, Training Q8 Accuracy: {avg_q8_training_acc:.4f}")

    # Validation loop
    model.eval()
    val_loss_sst3, val_loss_sst8 = 0, 0
    val_acc_sst3, val_acc_sst8 = 0, 0
    with torch.no_grad():
        for inputs, targets_sst3, targets_sst8 in test_loader:
            inputs, targets_sst3, targets_sst8 = inputs.to(device), targets_sst3.to(device), targets_sst8.to(device)

            outputs_sst3, outputs_sst8 = model(inputs)
            
            outputs_sst3 = outputs_sst3.permute(1, 2, 0)  #[batch_size, num_classes, seq_len]
            outputs_sst8 = outputs_sst8.permute(1, 2, 0)  #[batch_size, num_classes, seq_len]
            
            # print(outputs_sst3.shape)
            # print(targets_sst3.shape)

            # Calculate validation loss
            val_loss_sst3 += loss_sst3(outputs_sst3, targets_sst3).item()
            val_loss_sst8 += loss_sst8(outputs_sst8, targets_sst8).item()

            # Calculate batch accuracies
            batch_q3_acc = q3_acc(targets_sst3, outputs_sst3)
            batch_q8_acc = q3_acc(targets_sst8, outputs_sst8)

            val_acc_sst3 += batch_q3_acc.item()
            val_acc_sst8 += batch_q8_acc.item()

        # Calculate average validation loss and accuracy
        avg_val_loss_sst3 = val_loss_sst3 / len(test_loader)
        avg_val_loss_sst8 = val_loss_sst8 / len(test_loader)
        avg_val_loss = (val_loss_sst3 + val_loss_sst8) / len(test_loader)
        avg_val_q3_acc = val_acc_sst3 / len(test_loader)
        avg_val_q8_acc = val_acc_sst8 / len(test_loader)

        val_loss_sst3_history.append(avg_val_loss_sst3)
        val_loss_sst8_history.append(avg_val_loss_sst8)
        val_q3_acc_history.append(avg_val_q3_acc)
        val_q8_acc_history.append(avg_val_q8_acc)
        
        # Early Stopping based on validation loss improvement 
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve == patience:
            print(f"Early stopping at epoch {epoch+1}")
            plot_epoch = epoch+1
            break

        print(f"\nValidation Loss SST3: {avg_val_loss_sst3:.4f}, Validation Loss SST8: {avg_val_loss_sst8:.4f}")
        print(f"Validation Q3 Accuracy: {avg_val_q3_acc:.4f}, Validation Q8 Accuracy: {avg_val_q8_acc:.4f}\n")

# Plot training and validation loss and accuracy over epochs for SST3 and SST8 tasks
plot_training_history(
    train_loss_sst3_history, val_loss_sst3_history,
    train_loss_sst8_history, val_loss_sst8_history,
    train_q3_acc_history, val_q3_acc_history,
    train_q8_acc_history, val_q8_acc_history
)
