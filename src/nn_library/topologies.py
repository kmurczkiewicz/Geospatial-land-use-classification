from tensorflow.keras import layers, models, activations


def TEST_TOPOLOGY(input_shape, num_of_classes):
    """
    Function that initializes ANN (Artificial Neural Network) model using TEST_TOPOLOGY.
    Multidimensional input (image in numpy array type) is converted to one dimension. Only hidden dense layers.

    :param input_shape: input shape for the model
    :param num_of_classes: number of output classes for the model
    :return: ANN model
    """
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=input_shape))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(num_of_classes, activation='softmax'))

    return model


def topology_A(input_shape, num_of_classes):
    """
    Function that initializes Shallow ANN (Artificial Neural Network) model using A topology.
    Multidimensional input (image in numpy array type) is converted to one dimension. Only two hidden dense layers.

    :param input_shape: input shape for the model
    :param num_of_classes: number of output classes for the model
    :return: ANN model
    """
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=input_shape))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(num_of_classes, activation='softmax'))
    model.summary()

    return model


def topology_B(input_shape, num_of_classes):
    """
    Function that initializes Deep ANN (Artificial Neural Network) model using B topology.
    Multidimensional input (image in numpy array type) is converted to one dimension. Only hidden dense layers.

    :param input_shape: input shape for the model
    :param num_of_classes: number of output classes for the model
    :return: CNN model
    """
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=input_shape))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(num_of_classes, activation='softmax'))

    return model


def topology_C(input_shape, num_of_classes):
    """
    Function that initializes Shallow CNN (Convolutional Neural Network) model using C topology.

    :param input_shape: input shape for the model
    :param num_of_classes: number of output classes for the model
    :return: CNN model
    """
    cnn_model = models.Sequential()

    cnn_model.add(layers.InputLayer(input_shape=input_shape))
    cnn_model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    cnn_model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    cnn_model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    cnn_model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    # Without dropout, over-fitting is present
    cnn_model.add(layers.Dropout(0.3))
    cnn_model.add(layers.Flatten())
    cnn_model.add(layers.Dense(num_of_classes, activation='softmax'))
    cnn_model.summary()

    return cnn_model


def topology_D(input_shape, num_of_classes):
    """
    Function that initializes Deep CNN (Convolutional Neural Network) model using D topology.

    :param input_shape: input shape for the model
    :param num_of_classes: number of output classes for the model
    :return: CNN model
    """
    cnn_model = models.Sequential()
    # Block 1
    cnn_model.add(layers.InputLayer(input_shape=input_shape))
    cnn_model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    cnn_model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    cnn_model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    cnn_model.add(layers.SpatialDropout2D(0.2))
    cnn_model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Block 2
    cnn_model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    cnn_model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    cnn_model.add(layers.SpatialDropout2D(0.2))
    cnn_model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Block 3
    cnn_model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    cnn_model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    cnn_model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    cnn_model.add(layers.SpatialDropout2D(0.2))
    cnn_model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Block 4
    cnn_model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
    cnn_model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
    cnn_model.add(layers.SpatialDropout2D(0.2))
    cnn_model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Top
    cnn_model.add(layers.Flatten())
    cnn_model.add(layers.Dropout(0.3))
    cnn_model.add(layers.Dense(1024, activation='relu'))
    cnn_model.add(layers.Dropout(0.3))
    cnn_model.add(layers.Dense(1024, activation='relu'))
    cnn_model.add(layers.Dropout(0.3))
    cnn_model.add(layers.Dense(num_of_classes, activation='softmax'))
    cnn_model.summary()

    return cnn_model
