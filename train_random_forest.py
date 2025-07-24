#NonDeepLearning
import os
import time
import torch
import torch.nn as nn
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

from data_preprocess import get_data_loaders
from train_efficientnet import build_model  

# Constants
TRAIN_DIR = 'dataset/train'
VALID_DIR = 'dataset/validate'
TEST_DIR = 'dataset/test'
MODEL_DIR = "models"
CNN_MODEL = "best_efficientnet_model.pth"
MODEL_NAME = "rf_model.pkl"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_features(model, dataloader, device):
    """
    Extracts 1280-dim feature vectors from EfficientNet-B0 CNN before classification layer.
    """
    model.eval()
    features = []
    labels = []

    #Create feature extractor
    feature_extractor = nn.Sequential(
        model.features,
        model.avgpool,  
        nn.Flatten()    
    ).to(device)

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            batch_features = feature_extractor(inputs).cpu().numpy()
            features.append(batch_features)
            labels.append(targets.numpy())

    return np.vstack(features), np.concatenate(labels)


def train_random_forest(X_train, y_train):
    """
    Trains a Random Forest classifier on extracted CNN features.
    """
    print("Training Random Forest classifier")
    start_time = time.time()

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    print(f"Training completed in {time.time() - start_time:.2f} seconds")
    return clf


def main():
    print("Loading data")
    train_loader, _, _ = get_data_loaders(TRAIN_DIR, VALID_DIR, TEST_DIR)

    print("Loading pretrained CNN model")
    cnn_model = build_model(num_classes=4).to(DEVICE)

    model_file = os.path.join(MODEL_DIR, CNN_MODEL)
    cnn_model.load_state_dict(torch.load(model_file, map_location=DEVICE))
    cnn_model.eval()

    print("Extracting features from training set")
    X_train_features, y_train = extract_features(cnn_model, train_loader, DEVICE)
    print(f"Train features: {X_train_features.shape}, Labels: {y_train.shape}")

    rf_model = train_random_forest(X_train_features, y_train)

    model_save_path = os.path.join(MODEL_DIR, MODEL_NAME)
    joblib.dump(rf_model, model_save_path)
    print(f"Random Forest model saved to {model_save_path}")


if __name__ == "__main__":
    main()