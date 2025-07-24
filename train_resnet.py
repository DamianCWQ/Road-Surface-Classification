import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18, ResNet18_Weights
import matplotlib.pyplot as plt
import os
import random
import numpy as np

from data_preprocess import get_data_loaders

# Constants
TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/validate"
TEST_DIR = "dataset/test"
MODEL_DIR = "models"
MODEL_NAME = "resnet18_model.pth"

NUM_CLASSES = 4
LEARNING_RATE = 1e-4
EPOCHS = 50
PATIENCE = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Model setup
def build_model(num_classes=NUM_CLASSES, freeze=True):
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)

    # Optionally freeze all layers except final
    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    # Replace classifier
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )

    # Unfreeze final fc layer parameters
    for param in model.fc.parameters():
        param.requires_grad = True

    return model.to(DEVICE)

# Training loop
def train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LEARNING_RATE, patience=PATIENCE):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_loss = float('inf')
    wait = 0

    for epoch in range(epochs):
        model.train()
        train_loss, correct, total = 0.0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100 * correct / total

        val_loss, val_acc = validate_model(model, val_loader, criterion)
        scheduler.step(val_loss)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wait = 0
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_resnet18_model.pth"))
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    return history

# Validation loop
def validate_model(model, val_loader, criterion):
    model.eval()
    val_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_val_loss = val_loss / len(val_loader)
    val_acc = 100 * correct / total
    return avg_val_loss, val_acc

# Plotting
def plot_training_history(history, model_name, save_dir="evaluation_results"):
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title(f'{model_name} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    save_path = os.path.join(save_dir, f"{model_name.lower().replace(' ', '_')}_training_history.png")
    plt.savefig(save_path)
    print(f"Training history plot saved to {save_path}")
    plt.close()

# Main
def main():
    set_seed(11)
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Preparing data loaders...")
    train_loader, val_loader, _ = get_data_loaders(TRAIN_DIR, VAL_DIR, TEST_DIR, seed=11)

    print("Building ResNet18 model...")
    model = build_model()

    print("Training started...")
    history = train_model(model, train_loader, val_loader)
    print("Training completed.")

    # Save final model
    save_path = os.path.join(MODEL_DIR, MODEL_NAME)
    torch.save(model.state_dict(), save_path)
    print(f"Final model saved to {save_path}")

    # Plot history
    plot_training_history(history, "ResNet18 Model")

if __name__ == "__main__":
    main()
