import json
import pathlib
import os

import src.execution.executor_source


class Executor:
    """
    Main class for application execution.
    """
    def __init__(self, display):
        execution_params = src.execution.executor_source.init_executor()
        self.display = display
        self.DEFAULT_NETWORK_NAME = execution_params["DEFAULT_NETWORK_NAME"]
        self.PATHS = execution_params["PATHS"]
        self.NN_TOPOLOGIES = execution_params["NN_TOPOLOGIES"]

    def execute_data_analysis(self):
        """
        Execute data preparation and data analysis stage.
        """
        data_dict = src.execution.executor_source.stage_prepare_data(self.PATHS, read_head=False)
        src.execution.executor_source.stage_analyze_data(self.PATHS, data_dict, self.display)

    def execute_full_flow(self, topology, epochs, optimizer, loss_function, metrics, save_model):
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
        """
        # 1. Prepare test, train and validation data.
        data_dict = src.execution.executor_source.stage_prepare_data(self.PATHS, read_head=False)

        # 2. Load test, train and validation data into memory
        data = src.execution.executor_source.stage_load_data(self.PATHS, data_dict)

        # 3. Create network model and compile it
        cnn_model = src.execution.executor_source.stage_nn_init(
            nn_topology=self.NN_TOPOLOGIES[topology],
            input_shape=(64, 64, 3),
            optimizer=optimizer,
            loss_function=loss_function,
            metrics=metrics
        )

        # 4. Train the model
        src.execution.executor_source.stage_nn_train(cnn_model, data, epochs)

        # 5. Test the model
        src.execution.executor_source.stage_nn_test(cnn_model, data)

        # 6. Save the model if specified
        if not save_model:
            return
        src.execution.executor_source.stage_nn_save(
            self.PATHS["NETWORK_SAVE_DIR"],
            self.DEFAULT_NETWORK_NAME + "_" + topology,
            cnn_model
        )

    def execute_test_networks(self):
        # 1. Prepare test, train and validation data. Display the results.
        data_dict = src.execution.executor_source.stage_prepare_data(self.PATHS, False)

        # 2. Load test, train and validation data into memory
        data = src.execution.executor_source.stage_load_data(self.PATHS, data_dict)

        # 3. Test all saved networks in app
        src.execution.executor_source.stage_test_saved_networks(self.PATHS, data)

    def execute_analyze_networks(self):
        # 1. Analyze all saved networks from default app directory. Display the results.
        src.execution.executor_source.stage_analyze_saved_networks(self.PATHS)
