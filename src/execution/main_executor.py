import json
import pathlib
import os

import src.execution.base_executor

import tensorflow as tf


class MainExecutor(src.execution.base_executor.BaseExecutor):
    def __init__(self, display):
        super().__init__()
        self.display = display

    def execute_data_analysis(self):
        """
        Execute data preparation and data analysis stage.
        """
        data_dict = self.stage_prepare_data(read_head=False)
        self.stage_analyze_data(data_dict, self.display)

    def execute_full_flow(self, topology, epochs, optimizer, loss_function, metrics, save_model, plot_probability=True):
        """
        Execute all stages. Prepare load train, test and validation data into memory,
        initialize convolutional neural network with given topology, train and test the network.
        Save the model locally.
        :param topology: str network topology name
        :param epochs: int number of network training iterations
        :param optimizer: tf optimizer to be used for network compilation
        :param loss_function: tf loss function to be used for network compilation
        :param metrics: list of metrics to be measured for network
        :param save_model: bool defining if output model should be saved
        :param plot_probability: bool to define if class probability heatmap should be displayed
        """
        data_dict = self.stage_prepare_data(read_head=False)
        data = self.stage_load_data(data_dict)
        cnn_model = self.stage_nn_init(
            nn_topology=self.NN_TOPOLOGIES[topology],
            input_shape=(64, 64, 3),
            optimizer=optimizer,
            loss_function=loss_function,
            metrics=metrics
        )
        self.stage_nn_train(cnn_model, data, epochs)
        self.stage_nn_test(cnn_model, data, plot_probability)
        if not save_model:
            return
        self.stage_nn_save(
            self.PATHS["NETWORK_SAVE_DIR"],
            self.DEFAULT_NETWORK_NAME + "_" + topology,
            cnn_model
        )

    def execute_test_networks(self, networks_to_test=[], plot_probability=True):
        """
        Execute test networks stage. Test all networks saved in app default dir.
        :param networks_to_test: list of networks to be tested, if empty list is provided, test all saved networks
        :param plot_probability: bool to define if class probability heatmap should be displayed
        """
        data_dict = self.stage_prepare_data(read_head=False)
        data = self.stage_load_data(data_dict)
        self.stage_test_saved_networks(data, networks_to_test, plot_probability)

    def execute_analyze_networks(self):
        """
        Execute analyze networks stage. Analyze all networks saved in app default dir.
        """
        self.stage_analyze_saved_networks()
