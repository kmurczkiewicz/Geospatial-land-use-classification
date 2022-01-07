from tensorflow.keras import layers, models, activations


def topology_A(input_shape, num_of_classes):
    cnn_model = models.Sequential()
    cnn_model.add(layers.Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    cnn_model.add(layers.MaxPooling2D((2, 2)))
    cnn_model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    cnn_model.add(layers.MaxPooling2D((2, 2)))
    cnn_model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))

    cnn_model.add(layers.Flatten())
    cnn_model.add(layers.Dense(64, activation='relu'))
    # Without dropout, over-fitting is present
    cnn_model.add(layers.Dropout(0.3))
    cnn_model.add(layers.Dense(num_of_classes, activation='softmax'))

    return cnn_model


def topology_B(input_shape, num_of_classes):
    pass
