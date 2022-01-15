import pytest
import pandas as pd

import src.data.prepare
import src.execution.executor_source


def test_data_init():
    """
    Test data frames initialization.
    """
    execution_data = src.execution.executor_source.init_executor()
    expected_result = {
        "test_data_frame": pd.read_csv(execution_data["PATHS"]["TEST_CSV"], nrows=5).drop("Unnamed: 0", axis=1),
        "train_data_frame": pd.read_csv(execution_data["PATHS"]["TRAIN_CSV"], nrows=5).drop("Unnamed: 0", axis=1),
        "val_data_frame": pd.read_csv(execution_data["PATHS"]["VAL_CSV"], nrows=5).drop("Unnamed: 0", axis=1),
    }
    actual_result = src.data.prepare.data_init(execution_data["PATHS"], read_head=True)
    for key, value in expected_result.items():
        assert value.equals(actual_result[key])
