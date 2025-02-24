import matplotlib.pyplot as plt
import os

def plot_training_history(train_loss_sst3, val_loss_sst3, train_loss_sst8, val_loss_sst8,
                          train_acc_sst3, val_acc_sst3, train_acc_sst8, val_acc_sst8,
                          save_dir="plots"):
    # Plot training and validation loss and accuracy over epochs for SST3 and SST8 tasks
    os.makedirs(save_dir, exist_ok=True)

    # SST3 Loss
    plt.figure()
    plt.plot(train_loss_sst3, label="Train SST3 Loss")
    plt.plot(val_loss_sst3, label="Validation SST3 Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SST3 Training and Validation Loss")
    plt.legend()
    plt.savefig(f"{save_dir}/sst3_loss.png")
    
    # SST8 Loss
    plt.figure()
    plt.plot(train_loss_sst8, label="Train SST8 Loss")
    plt.plot(val_loss_sst8, label="Validation SST8 Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SST8 Training and Validation Loss")
    plt.legend()
    plt.savefig(f"{save_dir}/sst8_loss.png")
    
    # SST3 Accuracy
    plt.figure()
    plt.plot(train_acc_sst3, label="Train SST3 Q3 Accuracy")
    plt.plot(val_acc_sst3, label="Validation SST3 Q3 Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Q3 Accuracy")
    plt.title("SST3 Training and Validation Q3 Accuracy")
    plt.legend()
    plt.savefig(f"{save_dir}/sst3_accuracy.png")
    
    # SST8 Accuracy
    plt.figure()
    plt.plot(train_acc_sst8, label="Train SST8 Q8 Accuracy")
    plt.plot(val_acc_sst8, label="Validation SST8 Q8 Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Q8 Accuracy")
    plt.title("SST8 Training and Validation Q8 Accuracy")
    plt.legend()
    plt.savefig(f"{save_dir}/sst8_accuracy.png")

    plt.close('all')
    print(f"Plots saved in '{save_dir}/'")
