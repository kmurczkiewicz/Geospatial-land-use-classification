from tensorflow.keras import layers, models, activations


def TEST_TOPOLOGY(input_shape, num_of_classes):
    """
    Function that initializes MLP (Multilayer perceptron) model using TEST_TOPOLOGY.
    Multidimensional input (image in numpy array type) is converted to one dimension. Only hidden dense layers.

    :param input_shape: input shape for the model
    :param num_of_classes: number of output classes for the model
    :return: MLP model
    """
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=input_shape))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(num_of_classes, activation='softmax'))

    return model


def topology_A(input_shape, num_of_classes, layer_activation_function):
    """
    Function that initializes Shallow MLP (Multilayer perceptron) model using A topology.
    Multidimensional input (image in numpy array type) is converted to one dimension. Only two hidden dense layers.

    :param input_shape: input shape for the model
    :param num_of_classes: number of output classes for the model
    :param layer_activation_function: str name of nn layer activation function
    :return: Shallow MLP model
    """
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=input_shape))
    model.add(layers.Dense(256, activation=layer_activation_function))
    model.add(layers.Dense(num_of_classes, activation='softmax'))
    model.summary()

    return model


def topology_B(input_shape, num_of_classes, layer_activation_function):
    """
    Function that initializes Deep MLP (Multilayer perceptron) model using B topology.
    Multidimensional input (image in numpy array type) is converted to one dimension. Only hidden dense layers.

    :param input_shape: input shape for the model
    :param num_of_classes: number of output classes for the model
    :param layer_activation_function: str name of nn layer activation function
    :return: Deep MLP model
    """
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=input_shape))
    model.add(layers.Dense(256, activation=layer_activation_function))
    model.add(layers.Dense(256, activation=layer_activation_function))
    model.add(layers.Dense(256, activation=layer_activation_function))
    model.add(layers.Dense(256, activation=layer_activation_function))
    model.add(layers.Dense(256, activation=layer_activation_function))
    model.add(layers.Dense(256, activation=layer_activation_function))
    model.add(layers.Dense(num_of_classes, activation='softmax'))

    return model


def topology_C(input_shape, num_of_classes, layer_activation_function):
    """
    Function that initializes Shallow CNN (Convolutional Neural Network) model using C topology.

    :param input_shape: input shape for the model
    :param num_of_classes: number of output classes for the model
    :param layer_activation_function: str name of nn layer activation function
    :return: Shallow CNN model
    """
    cnn_model = models.Sequential()

    cnn_model.add(layers.InputLayer(input_shape=input_shape))
    # Block 1
    cnn_model.add(layers.Conv2D(32, (3, 3), padding='same', activation=layer_activation_function, input_shape=input_shape))
    cnn_model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    # Block 2
    cnn_model.add(layers.Conv2D(64, (3, 3), padding='same', activation=layer_activation_function))
    cnn_model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    cnn_model.add(layers.Dropout(0.3))
    # Fully connected top
    cnn_model.add(layers.Flatten())
    cnn_model.add(layers.Dense(num_of_classes, activation='softmax'))
    cnn_model.summary()

    return cnn_model


def topology_D(input_shape, num_of_classes, layer_activation_function):
    """
    Function that initializes Deep CNN (Convolutional Neural Network) model using D topology.

    :param input_shape: input shape for the model
    :param num_of_classes: number of output classes for the model
    :return: Deep CNN model
    """

    cnn_model = models.Sequential()

    # Block 1
    cnn_model.add(layers.InputLayer(input_shape=input_shape))
    cnn_model.add(layers.Conv2D(32, (3, 3), padding='same', activation=layer_activation_function))
    cnn_model.add(layers.Conv2D(32, (3, 3), padding='same', activation=layer_activation_function))
    cnn_model.add(layers.Conv2D(32, (3, 3), padding='same', activation=layer_activation_function))
    cnn_model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Block 2
    cnn_model.add(layers.Conv2D(64, (3, 3), padding='same', activation=layer_activation_function))
    cnn_model.add(layers.Conv2D(64, (3, 3), padding='same', activation=layer_activation_function))
    cnn_model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Block 3
    cnn_model.add(layers.Conv2D(128, (3, 3), padding='same', activation=layer_activation_function))
    cnn_model.add(layers.Conv2D(128, (3, 3), padding='same', activation=layer_activation_function))
    cnn_model.add(layers.Conv2D(128, (3, 3), padding='same', activation=layer_activation_function))
    cnn_model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Block 4
    cnn_model.add(layers.Conv2D(256, (3, 3), padding='same', activation=layer_activation_function))
    cnn_model.add(layers.Conv2D(256, (3, 3), padding='same', activation=layer_activation_function))
    cnn_model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Fully connected top
    cnn_model.add(layers.Flatten())
    # dropout
    cnn_model.add(layers.Dropout(0.45))
    # dense_units_1
    cnn_model.add(layers.Dense(512, activation=layer_activation_function))
    cnn_model.add(layers.Dense(num_of_classes, activation='softmax'))
    cnn_model.summary()

    return cnn_model
