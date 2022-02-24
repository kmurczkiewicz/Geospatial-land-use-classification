from kerastuner import HyperModel

from tensorflow import keras
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D
)

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
        return model.fit(
            *args,
            # batch_size=hp.Choice("batch_size", [32, 64, 128, 256]),
            batch_size=hp.Choice("batch_size", [32]),
            **kwargs,
    )

    def build(self, hp):
        model = keras.Sequential()

        # Block 1
        model.add(
            Conv2D(
                filters=32,
                kernel_size=(3,3),
                padding='same',
                activation='relu',
                input_shape=self.input_shape
            )
        )
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Block 2
        model.add(
            Conv2D(
                filters=64,
                kernel_size=(3,3),
                padding='same',
                activation='relu',
                input_shape=self.input_shape
            )
        )
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Dropout layer
        model.add(
            Dropout(rate=hp.Float(
                'dropout',
                min_value=0.0,
                max_value=0.5,
                default=0.25,
                step=0.05,
            ))
        )

        # Flatten layer
        model.add(Flatten())

        # Fully connected output layer
        model.add(Dense(self.num_classes, activation='softmax'))

        # Compile the model
        model.compile(
            optimizer=self.optimizer(
                hp.Float(
                    'learning_rate',
                    min_value=4e-4,
                    max_value=1e-2,
                    sampling='LOG',
                    default=1e-3
                )
            ),
            loss=self.loss_function,
            metrics=['accuracy']
        )
        return model

