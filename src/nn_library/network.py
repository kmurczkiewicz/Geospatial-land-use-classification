"""Imports"""
import datetime
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot

import src.helpers.print_extensions
import src.helpers.timer


class Neural_network:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = None
        self.training_history = None

    def compile(self, optimizer, loss_function, metrics):
        self.model.compile(optimizer, loss_function, metrics)

    def train_cnn_model(self, data: dict, epochs_num: int):
        self.training_history = self.model.fit(
            data["X_train"],
            data["y_train"],
            epochs=epochs_num,
            validation_data=(data["X_val"], data["y_val"])
        )

    def test_network(self, data: dict):
        test_loss, test_acc = self.model.evaluate(data["X_test"],  data["y_test"], verbose=2)
        print(f"Test accuracy: {test_acc}")
        return test_acc

    def init_network(self):
        cnn_model = models.Sequential()
        cnn_model.add(layers.Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=self.input_shape))
        cnn_model.add(layers.MaxPooling2D((2, 2)))
        cnn_model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
        cnn_model.add(layers.MaxPooling2D((2, 2)))
        cnn_model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))

        cnn_model.add(layers.Flatten())
        cnn_model.add(layers.Dense(64, activation='relu'))
        cnn_model.add(layers.Dense(10))

        self.model = cnn_model

    def plot_model_result(self):
        matplotlib.pyplot.plot(self.training_history.history['accuracy'], label='accuracy')
        matplotlib.pyplot.plot(self.training_history.history['val_accuracy'], label='val_accuracy')
        matplotlib.pyplot.xlabel('Epoch')
        matplotlib.pyplot.ylabel('Accuracy')
        matplotlib.pyplot.ylim([0.5, 1])
        matplotlib.pyplot.legend(loc='lower right')

    def save_model(self, name, directory):
        date_str = datetime.datetime.now().strftime("%H%M%d%m%y")
        tf.saved_model.save(self.model, f"{directory}\\{name}_{date_str}")