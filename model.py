import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import Dropout, Flatten, Dense, Input
from tensorflow.keras.regularizers import l2


def buildCNN(inputShape=(32, 32, 3), numClasses=10):

    model = Sequential([
        Input(shape=inputShape),

        # Block 1
        Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.2),

        # Block 2
        Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.3),

        # Block 3
        Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.4),

        # Fully connected head
        Flatten(),
        Dense(256, activation='relu', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(numClasses, activation='softmax'),

    ], name='CIFAR10CNN')

    return model


def compileModel(model, lr=1e-3):
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


if __name__ == '__main__':
    net = buildCNN()
    net = compileModel(net)
    net.summary()
