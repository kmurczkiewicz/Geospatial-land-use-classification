import datetime
import json
import os
import numpy as np
import tensorflow as tf
import pandas as pd

from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix
from IPython.display import display

import matplotlib.pyplot
import seaborn

import src.helpers.print_extensions
import src.helpers.timer
import src.nn_library.topologies


class Neural_network:
    """Class to unify and optimize creation of neural networks using different set of parameters."""

    def __init__(self,
                 nn_topology=src.nn_library.topologies.topology_A,
                 input_shape=(64, 64, 3)
                 ):
        self.topology = nn_topology
        self.input_shape = input_shape
        self.num_of_classes = 10

        self.optimizer = None
        self.loss_function = None
        self.layer_activation_function = None

        self.model = None
        self.training_history = None
        self.training_history_plots = {}
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

    def train_cnn_model(self, data: dict, epochs_num: int, checkpoint_filepath, batch_size):
        """
        Function to train network using given params.

        :param data: dict of test, train and validation data to be used in training
        :param epochs_num: number of training iterations
        :param checkpoint_filepath: str path to model checkpoint object
        :param batch_size: batch size
        """
        # Define checkpoint to obtain best model weights from training
        model_checkpoint_callback  = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            monitor="val_accuracy",
            mode='max',
            save_best_only=True
        )
        early_stopping_callback = tf.keras.callbacks.EarlyStopping('val_loss', patience=5)
        # Train the model
        self.training_history = self.model.fit(
            data["X_train"],
            data["y_train"],
            epochs=epochs_num,
            validation_data=(data["X_val"], data["y_val"]),
            batch_size=batch_size,
            shuffle=True,
            verbose=1,
            callbacks=[model_checkpoint_callback, early_stopping_callback]
        )
        # Load best weights into the model
        self.model.load_weights(checkpoint_filepath)

    def test_network(self, data: dict, label_map, plot_probability):
        """
        Function to test network on given data.

        :param data: dict of test, train and validation data to be used in training
        :param plot_probability: bool to define if class probability heatmap should be displayed
        :return: str network testing accuracy
        """
        # Test the network
        test_loss, test_acc = self.model.evaluate(data["X_test"],  data["y_test"], verbose=1)
        print(f"Test accuracy: {test_acc}")
        if not self.FTA and not self.FTL:
            self.FTA = test_acc
            self.FTL = test_loss
        # Get class prediction probabilities
        y_predicted = self.model.predict(data["X_test"])
        if not plot_probability:
            return test_acc
        self.plot_test_confusion_matrix(
            data["y_test"],
            y_predicted=np.argmax(y_predicted, axis=1),
            class_labels=[key for key, value in label_map.items()]
        )
        return test_acc

    def init_network(self, layer_activation_function):
        """
        Function to initialize network object topology.

        :param layer_activation_function: str name of nn layer activation function
        """
        self.layer_activation_function = layer_activation_function
        self.model = self.topology(
            self.input_shape,
            self.num_of_classes,
            self.layer_activation_function
        )

    def plot_test_confusion_matrix(self, y_true, y_predicted, class_labels):
        """
        Function to display confusion matrix for network class predictions

        :param y_true: original class labels
        :param y_predicted: class labels predicted by models
        :param class_labels: list of all class labels
        """
        confusion_matrix_data_frame = pd.DataFrame(
            confusion_matrix(y_true, y_predicted),
            class_labels,
            class_labels
        )
        matplotlib.pyplot.figure(num=3, figsize=(10, 10))
        seaborn.set(font_scale=1)
        seaborn.heatmap(
            confusion_matrix_data_frame,
            annot=True,
            cmap="Greens",
            annot_kws={"size": 9},
            fmt="g"
        )
        matplotlib.pyplot.ylabel("Label")
        matplotlib.pyplot.xlabel("Prediction")

        display(matplotlib.pyplot.figure(num=3))

        self.training_history_plots["prediction"] = matplotlib.pyplot.figure(num=3)
        matplotlib.pyplot.close()

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
        display(matplotlib.pyplot.figure(num=figure_num))
        if figure_num == 0:
            self.training_history_plots["acc"] = matplotlib.pyplot.figure(figure_num)
        else:
            self.training_history_plots["loss"] = matplotlib.pyplot.figure(figure_num)
        matplotlib.pyplot.close()

    def save_model(self, name, directory):
        """
        Function to save network in given directory with given name in .pb format, and
        .json file containing network details.

        :param name: network file name that will be saved
        :param directory: directory where network file will be saved
        """
        date_time = datetime.datetime.now()
        model_name = f"{name}_{date_time.strftime('%H%M%d%m%y')}"
        model_save_dir = os.path.join(directory, model_name)
        model_details = {
            "network_name" : model_name,
            "FTA"          : self.FTA,
            "FTL"          : self.FTL,
            "topology"     : self.topology.__name__,
            "optimizer"    : type(self.optimizer).__name__,
            "loss_function": type(self.loss_function).__name__,
            "activation"   : self.layer_activation_function,
            "created"      : date_time.strftime("%H:%M:%S, %d/%m/%Y"),
        }
        # Save network model
        self.model.save(model_save_dir)

        # Save .json network descriptor
        with open(os.path.join(model_save_dir, "network_details.json"), 'w') as file:
            json.dump(model_details, file)

        # Save training history plots
        if not self.training_history:
            return
        self.training_history_plots["acc"].savefig(os.path.join(model_save_dir, "train_acc_history.png"))
        self.training_history_plots["loss"].savefig(os.path.join(model_save_dir, "train_loss_history.png"))
        self.training_history_plots["prediction"].savefig(os.path.join(model_save_dir, "prediction_heatmap.png"))

    def single_class_prediction(self, input):
        """
        Function to perform prediction on given image.

        :param input: image form of numpy array
        :return: predicted class value
        """
        y_predicted = self.model.predict(input)
        class_predicted = np.argmax(y_predicted, axis=1)
        return class_predicted[0]
