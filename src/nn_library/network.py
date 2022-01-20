import datetime
import json
import os
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot

import src.helpers.print_extensions
import src.helpers.timer
import src.nn_library.topologies


class Neural_network:
    def __init__(self,
                 nn_topology=src.nn_library.topologies.topology_A,
                 input_shape=(64, 64, 3)
                 ):
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
            # batch_size=128,
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

    def train_cnn_model_advanced(self, data: dict, epochs_num: int):
        """
        Experimental function
        :param data: dict of test, train and validation data to be used in training
        :param epochs_num: number of training iterations
        """
        train_ds = tf.data.Dataset.from_tensor_slices((data["X_train"], data["y_train"])).shuffle(10000).batch(32)
        test_ds = tf.data.Dataset.from_tensor_slices((data["X_test"], data["y_test"])).batch(32)

        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        test_loss = tf.keras.metrics.Mean(name='test_loss')
        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
        for epoch in range(epochs_num):
            # Reset the metrics at the start of the next epoch
            train_loss.reset_states()
            train_accuracy.reset_states()
            test_loss.reset_states()
            test_accuracy.reset_states()

            for images, labels in train_ds:
                self.train_step(images, labels, train_loss, train_accuracy)

            for test_images, test_labels in test_ds:
                self.test_step(test_images, test_labels, test_loss, test_accuracy)

            print(
                f'Epoch {epoch + 1}, '
                f'Loss: {train_loss.result()}, '
                f'Accuracy: {train_accuracy.result() * 100}, '
                f'Test Loss: {test_loss.result()}, '
                f'Test Accuracy: {test_accuracy.result() * 100}'
            )

    @tf.function
    def train_step(self, images, labels, train_loss, train_accuracy):
        with tf.GradientTape() as tape:
            predictions = self.model(images, training=True)
            loss = self.loss_function(labels, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        train_loss(loss)
        train_accuracy(labels, predictions)

    @tf.function
    def test_step(self, images, labels, test_loss, test_accuracy):
        predictions = self.model(images, training=False)
        t_loss = self.loss_function(labels, predictions)
        test_loss(t_loss)
        test_accuracy(labels, predictions)
