import torch
import torch.nn as nn
import joblib
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score
from data_preprocess import get_data_loaders
from train_efficientnet import build_model as build_efficientnet
from train_mobilenet import build_model as build_mobilenet
from train_resnet import build_model as build_resnet
from train_random_forest import extract_features

#Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 4
SAVE_DIR = "evaluation_results"

EFFICIENTNET_MODEL_PATH = "models/best_efficientnet_model.pth"
MOBILENET_MODEL_PATH = "models/best_mobilenetv2_model.pth"
RESNET_MODEL_PATH = "models/best_resnet18_model.pth"
RF_MODEL_PATH = "models/rf_model.pkl"

TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/validate"
TEST_DIR = "dataset/test"
CLASS_NAMES = ['Cracked_road', 'Flooded_muddy', 'Good_condition', 'Potholes']

def evaluate_model_predictions(y_true, y_pred, model_name):
    print(f"\nEvaluation for {model_name}")
    cm = confusion_matrix(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))
    print(f"Cohen's Kappa Score: {kappa:.4f}")

    # Plot confusion matrix
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"{model_name}_confusion_matrix.png"))
    plt.close()

def get_dl_predictions(model, dataloader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(labels.numpy())
    return np.array(y_true), np.array(y_pred)

def get_rf_predictions(rf_model, dataloader, cnn_model):
    print("Extracting features for Random Forest evaluation")
    X_test, y_test = extract_features(cnn_model, dataloader, DEVICE)
    y_pred = rf_model.predict(X_test)
    return y_test, y_pred

def main():
    print("Loading test data...")
    _, _, test_loader = get_data_loaders(TRAIN_DIR, VAL_DIR, TEST_DIR)

    os.makedirs(SAVE_DIR, exist_ok=True)

    print("Loading EfficientNet model...")
    efficientnet_model = build_efficientnet(NUM_CLASSES)
    efficientnet_model.load_state_dict(torch.load(EFFICIENTNET_MODEL_PATH, map_location=DEVICE))

    print("Evaluating EfficientNet model...")
    y_true_efficientnet, y_pred_efficientnet = get_dl_predictions(efficientnet_model, test_loader)
    evaluate_model_predictions(y_true_efficientnet, y_pred_efficientnet, "EfficientNet")

    print("Loading MobileNet model...")
    mobilenet_model = build_mobilenet(NUM_CLASSES)
    mobilenet_model.load_state_dict(torch.load(MOBILENET_MODEL_PATH, map_location=DEVICE))

    print("Evaluating MobileNet model...")
    y_true_mobilenet, y_pred_mobilenet = get_dl_predictions(mobilenet_model, test_loader)
    evaluate_model_predictions(y_true_mobilenet, y_pred_mobilenet, "MobileNet")

    print("Loading ResNet model...")
    resnet_model = build_resnet(NUM_CLASSES)
    resnet_model.load_state_dict(torch.load(RESNET_MODEL_PATH, map_location=DEVICE))

    print("Evaluating ResNet model...")
    y_true_resnet, y_pred_resnet = get_dl_predictions(resnet_model, test_loader)
    evaluate_model_predictions(y_true_resnet, y_pred_resnet, "ResNet")

    print("Loading Random Forest model...")
    rf_model = joblib.load(RF_MODEL_PATH)

    print("Evaluating Random Forest model")
    y_true_rf, y_pred_rf = get_rf_predictions(rf_model, test_loader, efficientnet_model)
    evaluate_model_predictions(y_true_rf, y_pred_rf, "Random Forest")


if __name__ == "__main__":
    main()