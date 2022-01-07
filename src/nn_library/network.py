import datetime
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot

import src.helpers.print_extensions
import src.helpers.timer


class Neural_network:
    def __init__(self, nn_topology, input_shape):
        self.topology = nn_topology
        self.input_shape = input_shape
        self.num_of_classes = 10
        self.model = None
        self.training_history = None

    def compile(self, optimizer, loss_function, metrics):
        """
        Function to compile network using given params.
        :param optimizer: str optimizer name to be used in compilation
        :param loss_function: tf.keras.losses object to measure loss funtionc
        :param metrics: array of metrics to be measured
        """
        self.model.compile(optimizer, loss_function, metrics)

    def train_cnn_model(self, data: dict, epochs_num: int):
        """
        Function to train network using given params.
        :param data: dict of test, train and validation data to be used in training
        :param epochs_num: number of training iterations
        """
        self.training_history = self.model.fit(
            data["X_train"],
            data["y_train"],
            epochs=epochs_num,
            validation_data=(data["X_val"], data["y_val"])
        )

    def test_network(self, data: dict):
        """
        Function to test network on given data.
        :param data: dict of test, train and validation data to be used in training
        :return: str network testing accuracy
        """
        test_loss, test_acc = self.model.evaluate(data["X_test"],  data["y_test"], verbose=2)
        print(f"Test accuracy: {test_acc}")
        return test_acc

    def init_network(self):
        """
        Function to initialize network object topology.
        """
        self.model = self.topology(
            self.input_shape,
            self.num_of_classes
        )

    def plot_model_result(self):
        """
        Function to plot network accuracy and loss function value over training (epochs).
        """
        max_acc = max(self.training_history.history['accuracy'])
        max_loss = max(self.training_history.history['loss'])

        matplotlib.pyplot.plot(self.training_history.history['accuracy'], label='accuracy')
        matplotlib.pyplot.plot(self.training_history.history['val_accuracy'], label='val_accuracy')
        matplotlib.pyplot.plot(self.training_history.history['loss'], label='loss')
        matplotlib.pyplot.xlabel('Epoch')
        matplotlib.pyplot.ylabel('Accuracy')
        matplotlib.pyplot.ylim([0, max_acc if max_acc > max_loss else max_loss])
        matplotlib.pyplot.legend(loc='lower right')

    def save_model(self, name, directory):
        """
        Function to save network in given directory with given name in .pb format.
        :param name: network file name that will be saved
        :param directory: directory where network file will be saved
        """
        date_str = datetime.datetime.now().strftime("%H%M%d%m%y")
        self.model.save(f"{directory}\\{name}_{date_str}")
