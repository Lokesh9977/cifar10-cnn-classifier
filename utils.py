import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score


classNames = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

outputsDir = 'outputs'
plotsDir   = os.path.join('outputs', 'plots')
modelsDir  = 'models'


def loadData():
    print("Loading CIFAR-10 dataset...")

    (xTrainFull, yTrainFull), (xTest, yTestRaw) = cifar10.load_data()

    yTrainFull = yTrainFull.flatten()
    yTestRaw   = yTestRaw.flatten()

    cutoff    = int(0.1 * len(xTrainFull))
    xVal      = xTrainFull[:cutoff]
    yValRaw   = yTrainFull[:cutoff]
    xTrain    = xTrainFull[cutoff:]
    yTrainRaw = yTrainFull[cutoff:]

    xTrain = xTrain.astype('float32') / 255.0
    xVal   = xVal.astype('float32')   / 255.0
    xTest  = xTest.astype('float32')  / 255.0

    yTrain = to_categorical(yTrainRaw, 10)
    yVal   = to_categorical(yValRaw,   10)
    yTest  = to_categorical(yTestRaw,  10)

    print(f"Train: {xTrain.shape}   Val: {xVal.shape}   Test: {xTest.shape}")

    return xTrain, yTrain, xVal, yVal, xTest, yTest, yTrainRaw, yValRaw, yTestRaw


def getCallbacks(savePath):
    os.makedirs(outputsDir, exist_ok=True)

    return [
        ModelCheckpoint(filepath=savePath, monitor='val_accuracy', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
        CSVLogger(os.path.join(outputsDir, 'training_log.csv'))
    ]


def plotAccuracy(history):
    os.makedirs(plotsDir, exist_ok=True)
    savePath = os.path.join(outputsDir, 'accuracy.png')

    plt.figure(figsize=(9, 5))
    plt.plot(history.history['accuracy'],     label='Train Accuracy',      linewidth=2, color='#1f77b4')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2, color='#ff7f0e', linestyle='--')
    plt.title('Accuracy vs Epochs', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(savePath, dpi=150)
    plt.close()
    print(f"Accuracy plot saved to {savePath}")


def plotLoss(history):
    os.makedirs(plotsDir, exist_ok=True)
    savePath = os.path.join(outputsDir, 'loss.png')

    plt.figure(figsize=(9, 5))
    plt.plot(history.history['loss'],     label='Train Loss',      linewidth=2, color='#d62728')
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2, color='#9467bd', linestyle='--')
    plt.title('Loss vs Epochs', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(savePath, dpi=150)
    plt.close()
    print(f"Loss plot saved to {savePath}")


def plotConfusionMatrix(yTrue, yPred):
    os.makedirs(outputsDir, exist_ok=True)
    savePath = os.path.join(outputsDir, 'confusion_matrix.png')

    cm     = confusion_matrix(yTrue, yPred)
    cmNorm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cmNorm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=classNames, yticklabels=classNames, linewidths=0.5)
    plt.title('Normalised Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(savePath, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {savePath}")


def computeMetrics(yTrue, yPred):
    precision = precision_score(yTrue, yPred, average='weighted', zero_division=0)
    recall    = recall_score(yTrue, yPred,    average='weighted', zero_division=0)
    f1        = f1_score(yTrue, yPred,        average='weighted', zero_division=0)

    print("\n" + "=" * 50)
    print("  Evaluation Metrics")
    print("=" * 50)
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print("=" * 50)

    report = classification_report(yTrue, yPred, target_names=classNames, digits=4)
    print("\nPer-Class Report:\n")
    print(report)

    reportPath = os.path.join(outputsDir, 'classification_report.txt')
    os.makedirs(outputsDir, exist_ok=True)
    with open(reportPath, 'w') as f:
        f.write("Precision : " + str(round(precision, 4)) + "\n")
        f.write("Recall    : " + str(round(recall, 4))    + "\n")
        f.write("F1 Score  : " + str(round(f1, 4))        + "\n\n")
        f.write(report)

    print(f"Report saved to {reportPath}")
    return precision, recall, f1


def showSamplePredictions(model, xTest, yTestRaw, n=10):
    os.makedirs(plotsDir, exist_ok=True)
    savePath = os.path.join(plotsDir, 'sample_predictions.png')

    indices   = np.random.choice(len(xTest), n, replace=False)
    images    = xTest[indices]
    labels    = yTestRaw[indices]

    preds     = model.predict(images, verbose=0)
    predicted = np.argmax(preds, axis=1)

    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.8, rows * 3.0))
    axes = axes.flatten()

    for i, ax in enumerate(axes[:n]):
        ax.imshow(images[i])
        color = 'green' if predicted[i] == labels[i] else 'red'
        ax.set_title(
            f"True: {classNames[labels[i]]}\nPred: {classNames[predicted[i]]}",
            fontsize=8, color=color
        )
        ax.axis('off')

    for ax in axes[n:]:
        ax.axis('off')

    plt.suptitle('Sample Test Predictions', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(savePath, dpi=150)
    plt.close()
    print(f"Sample predictions saved to {savePath}")
