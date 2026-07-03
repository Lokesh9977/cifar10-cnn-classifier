# CIFAR-10 CNN Image Classifier

A convolutional neural network built with TensorFlow/Keras to classify images from the CIFAR-10 dataset into 10 categories. Achieves **87.43% test accuracy** using a custom architecture with batch normalization, dropout, and L2 regularization.

## Overview

This project implements an end-to-end image classification pipeline: data loading and preprocessing, model training with callbacks (early stopping, learning rate reduction, checkpointing), evaluation with detailed metrics, and inference on custom user-provided images.

## Dataset

- **CIFAR-10**: 60,000 32x32 color images across 10 classes
- Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- Split: 45,000 training / 5,000 validation / 10,000 test

## Model Architecture

A sequential CNN with:
- Multiple Conv2D blocks with increasing filter depth
- MaxPooling2D for spatial downsampling
- BatchNormalization for training stability
- Dropout and L2 regularization to prevent overfitting
- Dense layers with softmax output for 10-class classification

## Results

| Metric | Score |
|---|---|
| Test Accuracy | **87.43%** |
| Macro Avg Precision | 0.8742 |
| Macro Avg Recall | 0.8743 |
| Macro Avg F1-Score | 0.8737 |
| Weighted Avg F1-Score | 0.8737 |

### Sample Prediction

Tested on a custom (non-dataset) German Shepherd image:
- Predicted: dog
- Confidence: 95.57%

Tested on a custom horse image:
- Predicted: horse
- Confidence: 100.00%

![Sample Prediction](outputs/plots/custom_predictions.png)

### Training Curves

![Accuracy](outputs/accuracy.png)
![Loss](outputs/loss.png)

### Confusion Matrix

![Confusion Matrix](outputs/confusion_matrix.png)

## Project Structure

CIFAR10_CNN/
- dataset/
- models/cifar10_best.keras
- outputs/plots, accuracy.png, loss.png, confusion_matrix.png, classification_report.txt, results_summary.txt
- model.py
- train.py
- predict.py
- utils.py
- requirements.txt
- README.md

## How to Run

### 1. Clone the repository

    git clone https://github.com/Lokesh9977/cifar10-cnn-classifier.git
    cd cifar10-cnn-classifier

### 2. Set up the environment

    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt

### 3. Train the model

    python train.py

### 4. Run predictions on your own image

    python predict.py --image path\to\your\image.jpg

## Tech Stack

- Python 3.11
- TensorFlow / Keras
- NumPy, Pandas
- Matplotlib, Seaborn
- scikit-learn (metrics)

## Future Improvements

- Data augmentation (random flips, rotations, crops)
- Transfer learning with a pretrained backbone
- Deploy as an interactive Streamlit/Gradio web demo
- Hyperparameter tuning via Keras Tuner or Optuna

## Author

Lokesh — Integrated M.Tech, Computer Science (AI Specialization), VIT Bhopal University