#DeepLearning
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import matplotlib.pyplot as plt
import os
import random
import numpy as np

from data_preprocess import get_data_loaders

#Constants
TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/validate"
TEST_DIR = "dataset/test"
MODEL_DIR = "models"
MODEL_NAME = "mobilenetv2_model.pth"

NUM_CLASSES = 4  #good_condition, potholes, cracked_road, flooded_muddy
LEARNING_RATE = 1e-4
EPOCHS = 50
PATIENCE = 10  #number of epochs to wait for validation loss improvement before early stopping

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Fix all random seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#Model preparation
def build_model(num_classes=NUM_CLASSES, dropout_rate=0.2, freeze_layers=10):
    """
    Load MobileNetV2 and modify its final classification layer to fit the number of classess in dataset.
    """
    weights = MobileNet_V2_Weights.DEFAULT
    model = mobilenet_v2(weights=weights)

    # Freeze early layers
    for i, param in enumerate(model.features.parameters()):
        if i < freeze_layers:
            param.requires_grad = False
    

    last_channel = model.last_channel
        
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(last_channel, 512),
        nn.BatchNorm1d(512),
        nn.LeakyReLU(0.2),
        nn.Dropout(p=dropout_rate),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.LeakyReLU(0.2),
        nn.Dropout(p=dropout_rate/2),
        nn.Linear(256, num_classes)
    )
    
    return model.to(DEVICE)


#Training loop
def train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LEARNING_RATE, patience=PATIENCE):
    """
    Trains the model for a given number of epochs and returns training history.
    """
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.RMSprop(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5) # RMSprop optimizer with momentum

    # OneCycleLR scheduler for dynamic learning rate adjustment
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=lr,
        steps_per_epoch=len(train_loader),
        epochs=epochs
    )

    #Dictionary to store training and validation metrics
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    best_val_loss = float('inf')
    counter = 0

    #Training for specified number of epochs
    for epoch in range(epochs):
        model.train()
    
        running_loss = 0.0
        total, correct = 0, 0 

        #Iterate over training batches
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()  #Clear previous gradients
            outputs = model(inputs)  #Forward pass

            loss = criterion(outputs, labels)  #Compute loss
            loss.backward()  #Backward pass
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()  #Update weights     

            running_loss += loss.item()

            #Calculate train accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        #Average train metrics
        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total

        #Validation
        val_loss, val_accuracy = validate_model(model, val_loader, criterion)

        # Update learning rate
        scheduler.step()

        #Save history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_accuracy)
        history['val_acc'].append(val_accuracy)

        #Print progress 
        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Train Acc: {train_accuracy:.2f}% | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_accuracy:.2f}%")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            # Save best model
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_mobilenetv2_model.pth"))
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
                
    return history

#Validation
def validate_model(model, val_loader, criterion):
    """
    Evaluates model on validation data. Returns average loss and accuracy
    """

    model.eval()
    total, correct = 0, 0
    val_loss = 0.0

    #Disable gradient calculation
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    accuracy = 100 * correct / total
    return avg_val_loss, accuracy

#Plot training history
def plot_training_history(history, model_name, save_dir="evaluation_results"):
    """
    Plots and saves training and validation accuracy/loss curves.
    """
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(12, 4))

    #Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title(f'{model_name} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend(loc='lower right')

    #Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')

    plt.tight_layout()
    
    #Save to file
    save_path = os.path.join(save_dir, f"{model_name.lower().replace(' ', '_')}_training_history.png")
    plt.savefig(save_path)
    print(f"Training history plot saved to {save_path}")
    
    plt.close()


def main():
    #Set fixed seeds
    set_seed(11)

    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Preparing data loaders")
    train_loader, val_loader, _ = get_data_loaders(TRAIN_DIR, VAL_DIR, TEST_DIR, seed=11)

    print("Building model")
    model = build_model()

    print("Starting training")
    history = train_model(model, train_loader, val_loader)
    print("Training completed.")
    print("Best model saved to models\best_mobilenetv2_model.pth")

    #Save the model
    save_path = os.path.join(MODEL_DIR, MODEL_NAME)
    torch.save(model.state_dict(), save_path)
    print(f"Final model saved to {save_path}")

    #Plot training history
    plot_training_history(history, "MobileNetV2 Model")

if __name__ == "__main__":
    main()