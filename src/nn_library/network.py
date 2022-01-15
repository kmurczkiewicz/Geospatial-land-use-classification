import datetime
import json
import os
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

        self.optimizer = None
        self.loss_function = None

        self.model = None
        self.training_history = None

        # First Test Accuracy
        self.FTA = None
        # First Test Loss
        self.FTL = None

    def compile(self, optimizer, loss_function, metrics):
        """
        Function to compile network using given params.
        :param optimizer: str optimizer name to be used in compilation
        :param loss_function: tf.keras.losses object to measure loss funtionc
        :param metrics: array of metrics to be measured
        """
        self.loss_function = loss_function
        self.optimizer = optimizer
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
            validation_data=(data["X_val"], data["y_val"]),
            batch_size=128,
            shuffle=True,
            verbose=1
        )

    def test_network(self, data: dict):
        """
        Function to test network on given data.
        :param data: dict of test, train and validation data to be used in training
        :return: str network testing accuracy
        """
        test_loss, test_acc = self.model.evaluate(data["X_test"],  data["y_test"], verbose=2)
        print(f"Test accuracy: {test_acc}")
        if not self.FTA and not self.FTL:
            self.FTA = test_acc
            self.FTL = test_loss
        return test_acc

    def init_network(self):
        """
        Function to initialize network object topology.
        """
        self.model = self.topology(
            self.input_shape,
            self.num_of_classes
        )

    def plot_model_result(self, mode, figure_num):
        """
        Function to plot network accuracy and loss function value over training (epochs).
        :param mode: str plotting mode, could be 'accuracy' or 'loss'
        :param figure_num: int number defining which plot to use
        """
        matplotlib.pyplot.figure(figure_num)
        matplotlib.pyplot.plot(self.training_history.history[mode], label=mode)
        matplotlib.pyplot.plot(self.training_history.history["val_" + mode], label="val_" + mode)
        matplotlib.pyplot.xlabel('Epoch')
        matplotlib.pyplot.ylabel(mode)
        matplotlib.pyplot.ylim(
            [
                min(self.training_history.history[mode]),
                max(self.training_history.history[mode])
            ]
        )
        matplotlib.pyplot.legend(loc='lower right')

    def save_model(self, name, directory):
        """
        Function to save network in given directory with given name in .pb format, and
        .json file containing network details.
        :param name: network file name that will be saved
        :param directory: directory where network file will be saved
        """
        date_time = datetime.datetime.now()
        model_name = f"{name}_{date_time.strftime('%H%M%d%m%y')}"
        model_save_dir = f"{directory}\\{model_name}"
        model_details = {
            "network_name" : model_name,
            "FTA"          : self.FTA,
            "FTL"          : self.FTL,
            "topology"     : self.topology.__name__,
            "optimizer"    : type(self.optimizer).__name__,
            "loss_function": type(self.loss_function).__name__,
            "created"      : date_time.strftime("%H:%M:%S, %d/%m/%Y"),
        }
        self.model.save(model_save_dir)
        with open(os.path.join(model_save_dir, "network_details.json"), 'w') as file:
            json.dump(model_details, file)
