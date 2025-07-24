# Road-Surface-Classification
A machine learning project that classifies road surface conditions using multiple deep learning models and a traditional machine learning approach. The system can identify four types of road conditions: Good Condition, Potholes, Cracked Road, and Flooded/Muddy surfaces. This group project is contributed by 4 team members, who are Xavier Tiong, Kelvin Ng, Yek Nai Chong and Damian Choy.

## Project Overview
This project implements and compares multiple machine learning approaches for road surface classification:
- **EfficientNet** - Transfer learning with EfficientNet architecture
- **MobileNetV2** - Lightweight model optimised for mobile deployment
- **ResNet18** - Residual neural network for robust feature extraction
- **Random Forest** - Traditional machine learning classifier

The project includes a user-friendly Gradio web interface for real-time classification of road surface images.

## Project Structure

```
├── classifier.py              # Main classification interface with Gradio GUI
├── data_preprocess.py         # Data loading and preprocessing utilities
├── evaluate.py               # Model evaluation and metrics
├── train_efficientnet.py     # EfficientNet model training
├── train_mobilenet.py        # MobileNetV2 model training
├── train_resnet.py           # ResNet18 model training
├── train_random_forest.py    # Random Forest model training
├── dataset/                  # Dataset directory
│   ├── train/               # Training images
│   ├── validate/            # Validation images
│   └── test/                # Test images
├── models/                   # Trained model files
└── evaluation_results/       # Training plots and evaluation metrics
```

### Dataset Structure
Organize your dataset in the following structure:
```
dataset/
├── train/
│   ├── Cracked_road/
│   ├── Flooded_muddy/
│   ├── Good_condition/
│   └── Potholes/
├── validate/
│   └── [same structure as train]
└── test/
    └── [same structure as train]
```

## Usage
### Training Models
Train individual models using the respective training scripts:
```
# Train EfficientNet
python train_efficientnet.py

# Train MobileNetV2
python train_mobilenet.py

# Train ResNet18
python train_resnet.py

# Train Random Forest
python train_random_forest.py
```

### Running the Classification Interface
Launch the Gradio web interface for interactive classification:
```
python classifier.py
```

This will start a web interface where you can:
- Upload road surface images
- Select which model to use for classification
- View predictions and confidence scores

### Model Evaluation
Evaluate trained models:
```
python evaluate.py
```

## Model Architecture
### Deep Learning Models
- **EfficientNet**: Uses transfer learning with frozen feature extraction layers
- **MobileNetV2**: Lightweight architecture suitable for mobile deployment
- **ResNet18**: Residual connections for improved gradient flow

### Training Configuration
- **Learning Rate**: 1e-4
- **Epochs**: 50 (with early stopping)
- **Patience**: 10 epochs
- **Image Size**: 224x224 pixels
- **Batch Size**: Configured in [`data_preprocess.py`](data_preprocess.py)

### Data Preprocessing
- Resize to 224x224 pixels
- Center cropping
- Tensor conversion
- Normalization (as defined in [`classifier.py`](classifier.py))

## Model Performance

Training history plots and confusion matrices are automatically saved to the `evaluation_results/` directory:
- Training/validation accuracy curves
- Training/validation loss curves
- Confusion matrices for each model

## Key Features
- **Multi-model Comparison**: Compare performance across different architectures
- **Transfer Learning**: Leverages pre-trained weights for improved performance
- **Early Stopping**: Prevents overfitting with patience-based stopping
- **Reproducible Results**: Fixed random seeds for consistent training
- **Interactive GUI**: User-friendly Gradio interface
- **GPU Support**: Automatic CUDA detection and usage

## Class Labels
The system classifies road surfaces into four categories:
1. **Cracked_road** - Roads with visible cracks
2. **Flooded_muddy** - Waterlogged or muddy road surfaces
3. **Good_condition** - Well-maintained road surfaces
4. **Potholes** - Roads with potholes or significant damage

## Configuration
Key parameters can be modified in the respective training files:
- `LEARNING_RATE`: Optimizer learning rate
- `EPOCHS`: Maximum training epochs
- `PATIENCE`: Early stopping patience
- `NUM_CLASSES`: Number of classification classes (4)

## Output Files
- **Models**: Saved to `models/` directory
  - `best_efficientnet_model.pth`
  - `best_mobilenetv2_model.pth`
  - `best_resnet18_model.pth`
  - `rf_model.pkl`
- **Plots**: Saved to `evaluation_results/` directory
- **Training History**: Automatic plotting and saving
