from tensorflow.keras import layers, models, activations


def TEST_TOPOLOGY(input_shape, num_of_classes):
    """
    Function that initializes ANN (Artificial Neural Network) model.
    Multidimensional input (image in numpy array type) is converted to one dimension. Only hidden dense layers.
    :param input_shape: input shape for the model
    :param num_of_classes: number of output classes for the model
    :return: ANN model
    """
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=input_shape))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(num_of_classes, activation='softmax'))

    return model


def topology_A(input_shape, num_of_classes):
    """
    Function that initializes ANN (Artificial Neural Network) model.
    Multidimensional input (image in numpy array type) is converted to one dimension. Only hidden dense layers.
    :param input_shape: input shape for the model
    :param num_of_classes: number of output classes for the model
    :return: ANN model
    """
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=input_shape))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(num_of_classes, activation='softmax'))
    model.summary()

    return model


def topology_B(input_shape, num_of_classes):
    """
    Function that initializes CNN (Convolutional Neural Network) model.
    :param input_shape: input shape for the model
    :param num_of_classes: number of output classes for the model
    :return: CNN model
    """
    cnn_model = models.Sequential()
    cnn_model.add(layers.Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    cnn_model.add(layers.MaxPooling2D((2, 2)))
    cnn_model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    cnn_model.add(layers.MaxPooling2D((2, 2)))
    cnn_model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    # Without dropout, over-fitting is present
    cnn_model.add(layers.Flatten())
    cnn_model.add(layers.Dropout(0.3))
    cnn_model.add(layers.Dense(64, activation='relu'))
    cnn_model.add(layers.Dense(num_of_classes, activation='softmax'))
    cnn_model.summary()

    return cnn_model


def topology_C(input_shape, num_of_classes):
    """
    Function that initializes experimental network model.
    :param input_shape: input shape for the model
    :param num_of_classes: number of output classes for the model
    :return: Experimental neural network model
    """
    cnn_model = models.Sequential()
    cnn_model.add(layers.Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    cnn_model.add(layers.Conv2D(16, (3, 3), padding='same', activation='relu'))
    cnn_model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

    cnn_model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    cnn_model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

    cnn_model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    cnn_model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

    cnn_model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    cnn_model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

    cnn_model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
    cnn_model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
    cnn_model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

    cnn_model.add(layers.Conv2D(512, (3, 3), padding='same', activation='relu'))
    cnn_model.add(layers.Conv2D(512, (3, 3), padding='same', activation='relu'))
    cnn_model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

    cnn_model.add(layers.Flatten())
    cnn_model.add(layers.Dropout(0.3))
    cnn_model.add(layers.Dense(512, activation='relu'))
    cnn_model.add(layers.Dense(512, activation='relu'))
    cnn_model.add(layers.Dense(num_of_classes, activation='softmax'))
    cnn_model.summary()

    return cnn_model
