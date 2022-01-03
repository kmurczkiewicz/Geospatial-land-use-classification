import pandas as pd

import IPython.display
import src.helpers.print_extensions


def data_init(paths):
    """
    Function to read .csv data files into pandas data frames.
    returns: dict with test, train and validation data frames
    """
    return {
        "test_data_frame": pd.read_csv(paths["TEST_CSV"]).drop("Unnamed: 0", axis=1),
        "train_data_frame": pd.read_csv(paths["TRAIN_CSV"]).drop("Unnamed: 0", axis=1),
        "val_data_frame": pd.read_csv(paths["VAL_CSV"]).drop("Unnamed: 0", axis=1),
    }


def display_prepared_data(data_dict):
    for key, data_frame in data_dict.items():
        src.helpers.print_extensions.print_variable(key)
        IPython.display.display(data_frame.head(5))