import os
import sys
import argparse
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

from utils import classNames, plotsDir, modelsDir

defaultModel = os.path.join(modelsDir, 'cifar10_best.keras')
targetSize   = (32, 32)


def loadModel(path):
    if not os.path.exists(path):
        print(f"Model not found at: {path}")
        print("Please run train.py first.")
        sys.exit(1)
    print(f"Loading model from {path} ...")
    return tf.keras.models.load_model(path)


def prepareImage(imagePath):
    if not os.path.exists(imagePath):
        print(f"Image not found: {imagePath}")
        return None
    img = Image.open(imagePath).convert('RGB').resize(targetSize, Image.LANCZOS)
    arr = np.array(img, dtype='float32') / 255.0
    return np.expand_dims(arr, axis=0)


def predictImage(model, imageArray):
    probs      = model.predict(imageArray, verbose=0)[0]
    classIdx   = np.argmax(probs)
    className  = classNames[classIdx]
    confidence = probs[classIdx] * 100.0
    return className, confidence, probs


def saveResultGrid(results):
    os.makedirs(plotsDir, exist_ok=True)
    savePath = os.path.join(plotsDir, 'custom_predictions.png')

    n    = len(results)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 4.2), squeeze=False)

    for i, res in enumerate(results):
        ax  = axes[i // cols][i % cols]
        img = res['array'][0]

        ax.imshow(img)
        ax.set_title(
            f"Pred: {res['className']}\nConf: {res['confidence']:.1f}%",
            fontsize=9, fontweight='bold'
        )
        ax.axis('off')

        barAx = ax.inset_axes([0, -0.55, 1, 0.45])
        colors = ['#1f77b4'] * 10
        probsFlat = np.asarray(res['probs'], dtype=float).flatten()
        colors[np.argmax(probsFlat)] = '#d62728'
        yPos = np.arange(len(classNames))
        barAx.barh(yPos, probsFlat, color=colors, height=0.7)
        barAx.set_yticks(yPos)
        barAx.set_yticklabels(classNames)
        barAx.set_xlim(0, 1)
        barAx.set_xlabel('Probability', fontsize=7)
        barAx.tick_params(axis='y', labelsize=6)
        barAx.tick_params(axis='x', labelsize=6)

    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis('off')

    plt.suptitle('Custom Image Predictions', fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(savePath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPrediction grid saved to {savePath}")


def main():
    parser = argparse.ArgumentParser(description='Predict CIFAR-10 class for an image.')
    parser.add_argument('--image', nargs='+', required=True, help='Path to image file(s)')
    parser.add_argument('--model', default=defaultModel,    help='Path to saved model')
    args = parser.parse_args()

    model = loadModel(args.model)

    results = []
    print("\n" + "=" * 55)
    print("  Prediction Results")
    print("=" * 55)

    for imgPath in args.image:
        arr = prepareImage(imgPath)
        if arr is None:
            continue

        className, confidence, probs = predictImage(model, arr)

        print(f"\n  File       : {os.path.basename(imgPath)}")
        print(f"  Prediction : {className.upper()}")
        print(f"  Confidence : {confidence:.2f}%")
        print("  Top 3:")

        top3 = np.argsort(probs)[::-1][:3]
        for rank, idx in enumerate(top3, start=1):
            print(f"    {rank}. {classNames[idx]:<12}  {probs[idx] * 100:.2f}%")

        results.append({
            'path'      : imgPath,
            'array'     : arr,
            'className' : className,
            'confidence': confidence,
            'probs'     : probs
        })

    print("=" * 55)

    if results:
        saveResultGrid(results)
    else:
        print("No valid images found.")
        sys.exit(1)


main()
