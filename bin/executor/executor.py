import json
import pathlib
import os

import src.executor.execute


class Executor:
    def __init__(self):
        self.MAIN_PATH = os.path.dirname(pathlib.Path().resolve())
        self.DEFAULT_NETWORK_NAME = "small_cnn"

        self.PATHS = {
            "DATASET"          : os.path.join(self.MAIN_PATH, "artefacts/dataset"),
            "TEST_CSV"         : os.path.join(self.MAIN_PATH, "artefacts/dataset/test.csv"),
            "TRAIN_CSV"        : os.path.join(self.MAIN_PATH, "artefacts/dataset/train.csv"),
            "VAL_CSV"          : os.path.join(self.MAIN_PATH, "artefacts/dataset/validation.csv"),
            "LABEL_MAP_PATH"   : os.path.join(self.MAIN_PATH, "artefacts/dataset/label_map.json"),
            "NETWORK_SAVE_DIR" : os.path.join(self.MAIN_PATH, "artefacts/models_pb"),
            "LABEL_MAP"        : ""
        }

        with open(self.PATHS["LABEL_MAP_PATH"]) as json_file:
            self.PATHS["LABEL_MAP"] = json.load(json_file)

    def execute_data_analysis(self):
        # 1. Prepare test, train and validation data. Display the results
        data_dict = src.executor.execute.stage_preparation(self.PATHS)

        # 2. Analyze test, train and validation data
        src.executor.execute.stage_analyze_data(self.PATHS, data_dict)

    def execute_full_flow(self):
        # 1. Prepare test, train and validation data. Display the results
        data_dict = src.executor.execute.stage_preparation(self.PATHS)

        # 2. Load test, train and validation data into memory
        data = src.executor.execute.stage_load_data(self.PATHS, data_dict)

        # 3. Create simple cnn model and compile it
        cnn_model = src.executor.execute.stage_nn_init(input_shape=(64, 64, 3))

        # 4. Train the model
        src.executor.execute.stage_nn_train(cnn_model, data)

        # 5. Test the model
        src.executor.execute.stage_nn_test(cnn_model, data)

        # 6. Save the model
        src.executor.execute.stage_nn_save(
            self.PATHS["NETWORK_SAVE_DIR"],
            self.DEFAULT_NETWORK_NAME,
            cnn_model
        )
