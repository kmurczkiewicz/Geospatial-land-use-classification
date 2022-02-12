import json
import pathlib
import os

import tensorflow as tf
import pandas as pd
import PIL
import IPython.display
import numpy as np

import src.helpers.timer
import src.helpers.print_extensions

import src.data.analyze
import src.data.load
import src.data.prepare

import src.nn_library.network_analyzer
import src.nn_library.network
import src.nn_library.topologies

import src.client_use_case.sat_img_classifier


class BaseExecutor:
    """Base executor class defining partial execution stages"""

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
            "LABEL_MAP": os.path.join(self.MAIN_PATH, ""),
            "MODEL_CHECKPOINT_PATH": os.path.join(self.MAIN_PATH, "artefacts/model_checkpoint/model_weights.h5"),
            "SAT_IMG_PATH": os.path.join(self.MAIN_PATH, "artefacts/sat_images"),
            "SAT_TILES_PATH": os.path.join(self.MAIN_PATH, "artefacts/sat_images/tiles"),
            "SAT_MAP_TILES_PATH": os.path.join(self.MAIN_PATH, "artefacts/sat_images/tiles_map"),
        }

        with open(self.PATHS["LABEL_MAP_PATH"]) as json_file:
            self.PATHS["LABEL_MAP"] = json.load(json_file)

        self.NN_TOPOLOGIES = {
            "TEST": src.nn_library.topologies.TEST_TOPOLOGY,
            "A": src.nn_library.topologies.topology_A,
            "B": src.nn_library.topologies.topology_B,
            "C": src.nn_library.topologies.topology_C,
            "D": src.nn_library.topologies.topology_D
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

    def stage_test_saved_networks(self, data, networks_to_test, plot_probability):
        """
        Function to load and test all networks created by app.

        :param data: dict of test, train and validation data to be used in training
        :param networks_to_test: list of networks to be tested, if empty list is provided, test all saved networks
        :param plot_probability: bool to define if class probability heatmap should be displayed
        :return: dict of networks
        """
        timer = src.helpers.timer.Timer()
        src.helpers.print_extensions.print_title(f"{self._get_execution_num()}. Test networks")
        # If test filter is empty, test all networks
        if not networks_to_test:
            networks_to_test = os.listdir(self.PATHS["NETWORK_SAVE_DIR"])

        for network_name in filter(
            lambda x: x in os.listdir(self.PATHS["NETWORK_SAVE_DIR"]),
            networks_to_test
        ):
            timer.set_timer()
            src.helpers.print_extensions.print_subtitle(f"Testing: {network_name}")
            nn_network_obj = src.nn_library.network.Neural_network()
            nn_network_obj.model = tf.keras.models.load_model(
                os.path.join(self.PATHS["NETWORK_SAVE_DIR"], network_name)
            )
            nn_network_obj.test_network(data, self.PATHS["LABEL_MAP"], plot_probability)
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

    def stage_analyze_given_network(self, network_name, layer_num, image_path):
        """
        Function to analyze given network model. For given model filters shapes, filters
        sample for given layer and feature maps for each conv2d layer are displayed.

        :param network_name: str network name to be analyzed
        :param layer_num: int number of layer to be analyzed
        :param image_path: str path to image to analyze kernels and feature map
        """
        src.helpers.print_extensions.print_title(f"{self._get_execution_num()}. Analyze {network_name}")
        nn_analyzer = src.nn_library.network_analyzer.NetworkAnalyzer(
            network_path=os.path.join(self.PATHS["NETWORK_SAVE_DIR"], network_name),
            layer_num=layer_num,
            img_path=os.path.join(self.PATHS["DATASET"], image_path)
        )
        nn_analyzer.full_analysis()

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
        cnn_model.train_cnn_model(data, epochs, self.PATHS["MODEL_CHECKPOINT_PATH"])
        timer.stop_timer()
        cnn_model.plot_model_result("accuracy", 0)
        cnn_model.plot_model_result("loss", 1)

    def stage_nn_test(self, cnn_model: src.nn_library.network.Neural_network, data, plot_probability):
        """
        Execute network testing stage.

        :param cnn_model: object of type src.nn_library.network.Neural_network
        :param data: dict of test, train and validation data to be used in training
        :param plot_probability: bool to define if class probability heatmap should be displayed
        """
        timer = src.helpers.timer.Timer()
        timer.set_timer()
        src.helpers.print_extensions.print_title(f"{self._get_execution_num()}. Test the model")
        cnn_model.test_network(data, self.PATHS["LABEL_MAP"], plot_probability)
        timer.stop_timer()

    def stage_nn_save(self, save_dir, network_name, cnn_model: src.nn_library.network.Neural_network):
        """
        Execute save network stage. Save network in given dir with given name in .pb format.

        :param save_dir: str where network model will be saved in .pb format
        :param network_name: str name of saved network
        :param cnn_model: object of type src.nn_library.network.Neural_network
        """
        timer = src.helpers.timer.Timer()
        timer.set_timer()
        src.helpers.print_extensions.print_title(f"{self._get_execution_num()}. Save the model")
        cnn_model.save_model(network_name, save_dir)
        timer.stop_timer()

    def stage_run_sat_img_classifier(self, sat_img_list, network_name):
        """
        Initialize SatelliteImageClassifier object and perform land use classification on given list
        of satellite images.

        :param sat_img_list: list of str names of satellite images to be used
        :param network_name: str name of network to be used for land use classification on given sat image
        """
        timer = src.helpers.timer.Timer()
        timer.set_timer()
        sat_img_classifier = src.client_use_case.sat_img_classifier.SatelliteImageClassifier(
            self.PATHS,
            network_name,
            sat_img_list
        )
        sat_img_classifier.run_classification()
        timer.stop_timer()
