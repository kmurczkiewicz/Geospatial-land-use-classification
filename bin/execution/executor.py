import json
import pathlib
import os

import src.execution.executor_source
import src.nn_library.nn_topologies


class Executor:
    """
    Main class for application execution.
    """
    def __init__(self):
        self.MAIN_PATH = os.path.dirname(pathlib.Path().resolve())
        self.DEFAULT_NETWORK_NAME = "network"

        self.PATHS = {
            "DATASET"          : os.path.join(self.MAIN_PATH, "artefacts/dataset"),
            "TEST_CSV"         : os.path.join(self.MAIN_PATH, "artefacts/dataset/test.csv"),
            "TRAIN_CSV"        : os.path.join(self.MAIN_PATH, "artefacts/dataset/train.csv"),
            "VAL_CSV"          : os.path.join(self.MAIN_PATH, "artefacts/dataset/validation.csv"),
            "LABEL_MAP_PATH"   : os.path.join(self.MAIN_PATH, "artefacts/dataset/label_map.json"),
            "NETWORK_SAVE_DIR" : os.path.join(self.MAIN_PATH, "artefacts/models_pb"),
            "LABEL_MAP"        : os.path.join(self.MAIN_PATH, "")
        }

        with open(self.PATHS["LABEL_MAP_PATH"]) as json_file:
            self.PATHS["LABEL_MAP"] = json.load(json_file)

        self.NN_TOPOLOGIES = {
            "A" : src.nn_library.nn_topologies.topology_A
        }

    def execute_data_analysis(self):
        """
        Execute data preparation and data analysis stage.
        """
        data_dict = src.execution.executor_source.stage_prepare_data(self.PATHS)
        src.execution.executor_source.stage_analyze_data(self.PATHS, data_dict)

    def execute_full_flow(self, topology):
        """
        Execute all stages. Prepare load train, test and validation data into memory,
        initialize convolutional neural network with given topology, train and test the network.
        Save the model locally.
        :param topology: str network topology name
        """
        # 1. Prepare test, train and validation data. Display the results.
        data_dict = src.execution.executor_source.stage_prepare_data(self.PATHS)

        # 2. Load test, train and validation data into memory
        data = src.execution.executor_source.stage_load_data(self.PATHS, data_dict)

        # 3. Create simple cnn model and compile it
        cnn_model = src.execution.executor_source.stage_nn_init(
            nn_topology=self.NN_TOPOLOGIES[topology],
            input_shape=(64, 64, 3)
        )

        # 4. Train the model
        src.execution.executor_source.stage_nn_train(cnn_model, data)

        # 5. Test the model
        src.execution.executor_source.stage_nn_test(cnn_model, data)

        # 6. Save the model
        src.execution.executor_source.stage_nn_save(
            self.PATHS["NETWORK_SAVE_DIR"],
            self.DEFAULT_NETWORK_NAME + "_" + topology,
            cnn_model
        )
