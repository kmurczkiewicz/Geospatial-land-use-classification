import pytest
import pandas as pd

import src.data.prepare
import src.execution.main_executor


def test_data_init():
    """
    Test data frames initialization.
    """
    executor = src.execution.main_executor.MainExecutor(display=False)
    expected_result = {
        "test_data_frame": pd.read_csv(executor.PATHS["TEST_CSV"], nrows=5).drop("Unnamed: 0", axis=1),
        "train_data_frame": pd.read_csv(executor.PATHS["TRAIN_CSV"], nrows=5).drop("Unnamed: 0", axis=1),
        "val_data_frame": pd.read_csv(executor.PATHS["VAL_CSV"], nrows=5).drop("Unnamed: 0", axis=1),
    }
    actual_result = src.data.prepare.data_init(executor.PATHS, read_head=True)
    for key, value in expected_result.items():
        assert value.equals(actual_result[key])
