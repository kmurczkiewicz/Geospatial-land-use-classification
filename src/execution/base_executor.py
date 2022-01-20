import json
import pathlib
import os

import tensorflow as tf
import pandas as pd

import src.helpers.timer
import src.helpers.print_extensions

import src.data.analyze
import src.data.load
import src.data.prepare

import src.execution.executor_helpers

import src.nn_library.network
import src.nn_library.topologies


class BaseExecutor:
    def __init__(self):
        self.MAIN_PATH = os.path.dirname(pathlib.Path().resolve())
        self.DEFAULT_NETWORK_NAME = "network"
        self.execution_num = 0

        self.PATHS = {
            "DATASET": os.path.join(self.MAIN_PATH, "artefacts/dataset"),
            "TEST_CSV": os.path.join(self.MAIN_PATH, "artefacts/dataset/test.csv"),
            "TRAIN_CSV": os.path.join(self.MAIN_PATH, "artefacts/dataset/train.csv"),
            "VAL_CSV": os.path.join(self.MAIN_PATH, "artefacts/dataset/validation.csv"),
            "LABEL_MAP_PATH": os.path.join(self.MAIN_PATH, "artefacts/dataset/label_map.json"),
            "NETWORK_SAVE_DIR": os.path.join(self.MAIN_PATH, "artefacts/models_pb"),
            "LABEL_MAP": os.path.join(self.MAIN_PATH, "")
        }

        with open(self.PATHS["LABEL_MAP_PATH"]) as json_file:
            self.PATHS["LABEL_MAP"] = json.load(json_file)

        self.NN_TOPOLOGIES = {
            "TEST": src.nn_library.topologies.TEST_TOPOLOGY,
            "A": src.nn_library.topologies.topology_A,
            "B": src.nn_library.topologies.topology_B,
            "C": src.nn_library.topologies.topology_C
        }

    def _get_execution_num(self):
        self.execution_num += 1
        return self.execution_num

    def stage_prepare_data(self, read_head):
        """
        Execute data preparation stage.
        Load test, train and validation .csv files into pandas data frames.
        :param read_head: bool, read only 5 rows from test, train and val data frame
        :return dict of train, test, validation pandas data frames
        """
        timer = src.helpers.timer.Timer()
        src.helpers.print_extensions.print_title(f"{self._get_execution_num()}. Prepare test, train and validation data")
        timer.set_timer()
        data_dict = src.data.prepare.data_init(self.PATHS, read_head)
        src.data.prepare.display_prepared_data(data_dict)
        timer.stop_timer()
        src.helpers.print_extensions.print_border()
        return data_dict

    def stage_analyze_data(self, data_dict, display):
        """
        Execute analyze data stage.
        Display label map, amount of class labels in test, train and validation data frame,
        and plot 5x5 images from each dataframe.
        :param data_dict: dict of train, test, validation pandas data frames
        :param display: bool type, display output (images, plots etc.)
        """
        timer = src.helpers.timer.Timer()
        timer.set_timer()
        src.helpers.print_extensions.print_title(f"{self._get_execution_num()}. Analyze test, train and validation data")
        src.data.analyze.analyze_data(self.PATHS, data_dict, display)
        timer.stop_timer()

    def stage_load_data(self, data_dict):
        """
        Execute load data stage.
        Load test, train and validation data into memory.
        :param data_dict: dict of train, test, validation pandas data frames
        :return: dict of test, train and validation data
        """
        timer = src.helpers.timer.Timer()
        timer.set_timer()
        src.helpers.print_extensions.print_title(
            f"{self._get_execution_num()}. Load test, train and val data into memory"
        )
        data = src.data.load.load_into_memory(self.PATHS, data_dict)
        timer.stop_timer()
        src.helpers.print_extensions.print_border()
        return data

    def stage_test_saved_networks(self, data):
        """
        Function to load and test all networks created by app.
        :param data: dict of test, train and validation data to be used in training
        :return: dict of networks
        """
        timer = src.helpers.timer.Timer()
        src.helpers.print_extensions.print_title(f"{self._get_execution_num()}. Test all saved networks")
        timer.set_timer()
        src.execution.executor_helpers.test_saved_networks(self.PATHS, data)
        timer.stop_timer()

    def stage_analyze_saved_networks(self):
        """
        Function to load and display analysis all networks created by app.
        """
        timer = src.helpers.timer.Timer()
        timer.set_timer()
        src.helpers.print_extensions.print_title(f"{self._get_execution_num()}. Analyze all saved networks")
        src.data.analyze.analyze_saved_networks(self.PATHS)
        timer.stop_timer()

    def stage_nn_init(
            self,
            nn_topology,
            input_shape,
            optimizer,
            loss_function,
            metrics,
    ):
        """
        Execute neural network initialization stage.
        Create network object and compile the network.
        :param nn_topology: str topology name to be used
        :param input_shape: tuple of three integers
        :param optimizer: tf optimizer to be used for network compilation
        :param loss_function: tf loss function to be used for network compilation
        :param metrics: list of metrics to be measured for network
        :return: initialized network model
        """
        timer = src.helpers.timer.Timer()
        timer.set_timer()
        src.helpers.print_extensions.print_title(f"{self._get_execution_num()}. Create and compile the model")
        cnn_model = src.nn_library.network.Neural_network(nn_topology, input_shape)
        cnn_model.init_network()
        cnn_model.compile(optimizer, loss_function, metrics)
        timer.stop_timer()
        src.helpers.print_extensions.print_border()
        return cnn_model

    def stage_nn_train(self, cnn_model: src.nn_library.network.Neural_network, data, epochs):
        """
        Execute network training stage.
        :param cnn_model: object of type src.nn_library.network.Neural_network
        :param data: dict of test, train and validation data to be used in training
        :param epochs: number of training iterations
        """
        timer = src.helpers.timer.Timer()
        src.helpers.print_extensions.print_title(f"{self._get_execution_num()}. Train the model")
        timer.set_timer()
        cnn_model.train_cnn_model(data, epochs)
        timer.stop_timer()
        cnn_model.plot_model_result("accuracy", 0)
        cnn_model.plot_model_result("loss", 1)

    def stage_nn_test(self, cnn_model: src.nn_library.network.Neural_network, data):
        """
        Execute network testing stage.
        :param cnn_model: object of type src.nn_library.network.Neural_network
        :param data: dict of test, train and validation data to be used in training
        """
        src.helpers.print_extensions.print_title(f"{self._get_execution_num()}. Test the model")
        cnn_model.test_network(data)

    def stage_nn_save(self, save_dir, network_name, cnn_model: src.nn_library.network.Neural_network):
        """
        Execute save network stage. Save network in given dir with given name in .pb format.
        :param save_dir: str where network model will be saved in .pb format
        :param network_name: str name of saved network
        :param cnn_model: object of type src.nn_library.network.Neural_network
        """
        src.helpers.print_extensions.print_title(f"{self._get_execution_num()}. Save the model")
        cnn_model.save_model(network_name, save_dir)
