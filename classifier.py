import os
import torch
import torch.nn as nn
import numpy as np
import joblib
import gradio as gr
from torchvision import transforms
from train_efficientnet import build_model as build_efficientnet 
from train_mobilenet import build_model as build_mobilenet
from train_resnet import build_model as build_resnet

#Constants
CLASS_NAMES = ['Cracked_road', 'Flooded_muddy', 'Good_condition', 'Potholes']
EFFICIENTNET_MODEL_PATH = "models/best_efficientnet_model.pth"
MOBILENET_MODEL_PATH = "models/best_mobilenetv2_model.pth"
RESNET_MODEL_PATH = "models/best_resnet18_model.pth"
RF_MODEL_PATH = "models/rf_model.pkl"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Image transformation same as validate and test dataset
transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


def load_models():
    """
    Load EfficientNet, MobileNet and Random Forest models.
    """
    # Load EfficientNet model
    efficientnet = build_efficientnet()
    efficientnet.load_state_dict(torch.load(EFFICIENTNET_MODEL_PATH, map_location=DEVICE))
    efficientnet.to(DEVICE)
    efficientnet.eval()
    
    # Load MobileNet model
    mobilenet = build_mobilenet()
    mobilenet.load_state_dict(torch.load(MOBILENET_MODEL_PATH, map_location=DEVICE))
    mobilenet.to(DEVICE)
    mobilenet.eval()

    # Load ResNet model
    resnet = build_resnet()
    resnet.load_state_dict(torch.load(RESNET_MODEL_PATH, map_location=DEVICE))
    resnet.to(DEVICE)
    resnet.eval()

    # Load Random Forest model
    rf = joblib.load(RF_MODEL_PATH)

    return efficientnet, mobilenet, resnet, rf


def extract_single_feature(model, input_tensor, device):
    """
    Extract a single 1280-dim feature vector from EfficientNet-B0 CNN for Random Forest input.
    """
    model.eval()
    feature_extractor = nn.Sequential(
        model.features,
        model.avgpool,
        nn.Flatten()
    ).to(device)

    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        features = feature_extractor(input_tensor).cpu().numpy()
    return features


#Prediction
def predict(img, model_choice, efficientnet_model, mobilenet_model, resnet_model, rf_model):
    """
    Predict the class label and confidence scores for the input image using selected model.
    """
    if img is None:
        return "No image provided", ""
    
    try:
        image = img.convert("RGB")
        tensor = transform(image).unsqueeze(0)
    except Exception as e:
        return "Error processing image", str(e)

    if model_choice == "EfficientNet":
        try:
            with torch.no_grad():
                outputs = efficientnet_model(tensor.to(DEVICE))
                probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        except Exception as e:
            return "EfficientNet prediction error", str(e)
    
    elif model_choice == "MobileNet":
        try:
            with torch.no_grad():
                outputs = mobilenet_model(tensor.to(DEVICE))
                probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        except Exception as e:
            return "MobileNet prediction error", str(e)
        
    elif model_choice == "ResNet":
        try:
            with torch.no_grad():
                outputs = resnet_model(tensor.to(DEVICE))
                probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        except Exception as e:
            return "ResNet prediction error", str(e)
    
    else:  # Random Forest
        try:
            features = extract_single_feature(efficientnet_model, tensor, DEVICE)
            probs = rf_model.predict_proba(features)[0]
        except Exception as e:
            return "Random Forest prediction error", str(e)
        
    predicted_idx = int(np.argmax(probs))
    label = CLASS_NAMES[predicted_idx]
    details = "\n".join(f"{CLASS_NAMES[i]}: {probs[i]:.4f}" for i in range(len(CLASS_NAMES)))
    return label, details

#Create GUI
def create_gui(efficientnet_model, mobilenet_model, resnet_model, rf_model):
    """
    Set up and launch the Gradio interface.
    """
    with gr.Blocks() as demo:
        gr.Markdown("## Road Surface Classification")

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Input Image")
                model_selector = gr.Radio(
                    choices=["EfficientNet", "MobileNet", "ResNet","Random Forest"], 
                    value="EfficientNet", 
                    label="Select Model"
                )
                classify_btn = gr.Button("Classify")
            with gr.Column():
                label_output = gr.Textbox(label="Predicted Class")
                confidence_output = gr.Textbox(label="Confidence Scores", lines=5)

        classify_btn.click(
            fn=lambda img, model_choice: predict(img, model_choice, efficientnet_model, mobilenet_model, resnet_model, rf_model),
            inputs=[image_input, model_selector],
            outputs=[label_output, confidence_output]
        )
        
        demo.launch(share=False, inbrowser=True)


def main():
    # Check model files exist
    if not os.path.isfile(EFFICIENTNET_MODEL_PATH):
        raise FileNotFoundError(f"Missing EfficientNet model file: {EFFICIENTNET_MODEL_PATH}")
    if not os.path.isfile(MOBILENET_MODEL_PATH):
        raise FileNotFoundError(f"Missing MobileNet model file: {MOBILENET_MODEL_PATH}")
    if not os.path.isfile(RESNET_MODEL_PATH):
        raise FileNotFoundError(f"Missing ResNet model file: {RESNET_MODEL_PATH}")
    if not os.path.isfile(RF_MODEL_PATH):
        raise FileNotFoundError(f"Missing Random Forest model file: {RF_MODEL_PATH}")

    print("Loading models")
    efficientnet_model, mobilenet_model, resnet_model, rf_model = load_models()

    print("Models loaded successfully.")

    print("Launching GUI")
    create_gui(efficientnet_model, mobilenet_model, resnet_model, rf_model)


if __name__ == "__main__":
    main()