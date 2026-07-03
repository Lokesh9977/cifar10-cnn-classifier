import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

from model import buildCNN, compileModel
from utils import (
    loadData,
    getCallbacks,
    plotAccuracy,
    plotLoss,
    plotConfusionMatrix,
    computeMetrics,
    showSamplePredictions,
    outputsDir,
    modelsDir
)

epochs    = 50
batchSize = 64
lr        = 1e-3
modelFile = os.path.join(modelsDir, 'cifar10_best.keras')


def main():
    os.makedirs(outputsDir, exist_ok=True)
    os.makedirs(modelsDir,  exist_ok=True)

    xTrain, yTrain, xVal, yVal, xTest, yTest, yTrainRaw, yValRaw, yTestRaw = loadData()

    print("\nBuilding model...")
    net = buildCNN(inputShape=(32, 32, 3), numClasses=10)
    net = compileModel(net, lr=lr)
    net.summary()

    callbacks = getCallbacks(modelFile)

    print(f"\nStarting training for up to {epochs} epochs...")
    history = net.fit(
        xTrain, yTrain,
        epochs          = epochs,
        batch_size      = batchSize,
        validation_data = (xVal, yVal),
        callbacks       = callbacks,
        verbose         = 1
    )

    print(f"\nLoading best saved model from {modelFile}")
    bestModel = tf.keras.models.load_model(modelFile)

    trainLoss, trainAcc = bestModel.evaluate(xTrain, yTrain, verbose=0)
    valLoss,   valAcc   = bestModel.evaluate(xVal,   yVal,   verbose=0)
    testLoss,  testAcc  = bestModel.evaluate(xTest,  yTest,  verbose=0)

    print("\n" + "=" * 50)
    print("  Accuracy Summary")
    print("=" * 50)
    print(f"  Training   : {trainAcc * 100:.2f}%  (loss: {trainLoss:.4f})")
    print(f"  Validation : {valAcc   * 100:.2f}%  (loss: {valLoss:.4f})")
    print(f"  Test       : {testAcc  * 100:.2f}%  (loss: {testLoss:.4f})")
    print("=" * 50)

    print("\nRunning predictions on test set...")
    preds      = bestModel.predict(xTest, batch_size=batchSize, verbose=1)
    predLabels = np.argmax(preds, axis=1)

    precision, recall, f1 = computeMetrics(yTestRaw, predLabels)

    plotAccuracy(history)
    plotLoss(history)
    plotConfusionMatrix(yTestRaw, predLabels)
    showSamplePredictions(bestModel, xTest, yTestRaw, n=10)

    summaryPath = os.path.join(outputsDir, 'results_summary.txt')
    with open(summaryPath, 'w') as f:
        f.write("CIFAR-10 CNN Results\n")
        f.write("=" * 40 + "\n")
        f.write(f"Training   Accuracy : {trainAcc * 100:.2f}%\n")
        f.write(f"Validation Accuracy : {valAcc   * 100:.2f}%\n")
        f.write(f"Test       Accuracy : {testAcc  * 100:.2f}%\n")
        f.write(f"Precision           : {precision:.4f}\n")
        f.write(f"Recall              : {recall:.4f}\n")
        f.write(f"F1 Score            : {f1:.4f}\n")

    print(f"\nResults saved to {summaryPath}")
    print(f"Model saved  to {modelFile}")
    print("\nDone. Check the outputs folder for all plots and reports.")


main()
