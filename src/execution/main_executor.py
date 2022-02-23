import json
import pathlib
import os

import src.execution.base_executor

import tensorflow as tf


class MainExecutor(src.execution.base_executor.BaseExecutor):
    """
    Main executor class that defines execution blocks to perform complex tasks using partial stages
    from parent BaseExecutor class.
    """

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

    def execute_analyze_single_network(self, network_name, layer_num, image_path):
        """
        Execute analyze given network stage.

        :param network_name: str network name to be analyzed
        :param layer_num: int number of layer to be analyzed
        :param image_path: str path to image to analyze kernels and feature map
        """
        self.stage_analyze_given_network(network_name, layer_num, image_path)

    def execute_land_use_classification_use_case(self, sat_img_list, network_name):
        """
        Execute land use classification use case stage. Load satellite image, split it into tiles
        and perform classification with given network on each tile. Further, land use classification map
        is generated based on networks' predictions.

        :param sat_img_list: list of str names of satellite images to be used
        :param network_name: str name of network to be used for land use classification on given sat image
        """
        self.stage_run_sat_img_classifier(
            sat_img_list,
            network_name
        )

    def execute_nn_hyper_parameters_tuning(
            self,
            overwrite,
            max_trials,
            executions_per_trial,
            n_epoch_search
    ):
        """
        Execute neural network hyper parameters tuning stage.

        :param overwrite: bool, overwrite existing tuner
        :param max_trials: int, num of max trials for tuning
        :param executions_per_trial: int, num of executions per trial
        :param n_epoch_search: int, num of epochs to perform search
        """
        data_dict = self.stage_prepare_data(read_head=False)
        data = self.stage_load_data(data_dict)
        self.stage_hyper_parameters_tuning(
            data,
            overwrite,
            max_trials,
            executions_per_trial,
            n_epoch_search,
        )
