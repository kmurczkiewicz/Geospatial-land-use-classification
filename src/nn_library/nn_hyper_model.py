from kerastuner import HyperModel

from tensorflow import keras
from tensorflow.keras import layers


class NetworkHyperModel(HyperModel):
    """
    Class to manage Keras Hyper Model structure.
    """

    def __init__(self):
        self.input_shape = (64, 64, 3)
        self.num_classes = 10
        self.optimizer = keras.optimizers.Adam
        self.loss_function = "sparse_categorical_crossentropy"

    def fit(self, hp, model, *args, **kwargs):
        """
        Function to overwrite tensorflow fit function.

        :param hp: HyperParameters object
        :param model: tensorflow model object
        :param args: *args
        :param kwargs: **kwargs
        :return: overwritten tensorflow model.fit function
        """
        return model.fit(
            *args,
            batch_size=hp.Choice("batch_size", [32, 64, 128, 256]),
            **kwargs,
    )

    def build(self, hp):
        """
        Function to define keras hyper model, a model which hyper-parameters
        are defined by ex. list of variables. Which are further used in
        hyper-parameters tuning stage.

        :param hp: HyperParameters object
        :return: HyperModel object
        """
        model = keras.Sequential()

        # Block 1
        model.add(layers.InputLayer(input_shape=self.input_shape))

        model.add(
            layers.Conv2D(
                filters=32,
                kernel_size=(3,3),
                padding='same',
                activation='relu',
            )
        )
        model.add(
            layers.Conv2D(
                filters=32,
                kernel_size=(3,3),
                padding='same',
                activation='relu',
            )
        )
        model.add(
            layers.Conv2D(
                filters=32,
                kernel_size=(3,3),
                padding='same',
                activation='relu',
            )
        )

        model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # Block 2
        model.add(
            layers.Conv2D(
                filters=64,
                kernel_size=(3,3),
                padding='same',
                activation='relu',
            )
        )
        model.add(
            layers.Conv2D(
                filters=64,
                kernel_size=(3,3),
                padding='same',
                activation='relu',
            )
        )

        model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # Block 3
        model.add(
            layers.Conv2D(
                filters=128,
                kernel_size=(3,3),
                padding='same',
                activation='relu',
            )
        )
        model.add(
            layers.Conv2D(
                filters=128,
                kernel_size=(3,3),
                padding='same',
                activation='relu',
            )
        )
        model.add(
            layers.Conv2D(
                filters=128,
                kernel_size=(3,3),
                padding='same',
                activation='relu',
            )
        )

        model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # Block 4
        model.add(
            layers.Conv2D(
                filters=256,
                kernel_size=(3,3),
                padding='same',
                activation='relu',
            )
        )
        model.add(
            layers.Conv2D(
                filters=256,
                kernel_size=(3,3),
                padding='same',
                activation='relu',
            )
        )

        model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # Flatten and dropout layers
        model.add(layers.Flatten())
        model.add(
            layers.Dropout(rate=hp.Float(
                'dropout',
                min_value=0.0,
                max_value=0.5,
                default=0.3,
                step=0.05,
            ))
        )
        # Fully connected output layer
        model.add(layers.Dense(
            units=hp.Choice("dense_units_1", [256, 512, 1024, 2048]),
            activation='relu')
        )
        model.add(layers.Dense(self.num_classes, activation='softmax'))

        # Compile the model
        model.compile(
            optimizer=self.optimizer(
                hp.Float(
                    'learning_rate',
                    min_value=1e-4,
                    max_value=5e-4,
                    sampling='LOG',
                    default=3e-4
                )
            ),
            loss=self.loss_function,
            metrics=['accuracy']
        )
        return model
