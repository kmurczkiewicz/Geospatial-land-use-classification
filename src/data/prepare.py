import pandas as pd
import numpy as np

import IPython.display
import PIL.Image

import src.helpers.print_extensions


def data_init(paths, read_head):
    """
    Function to read .csv data files into pandas data frames.
    :param paths: dict of app paths
    :param read_head: bool, read only 5 rows from test, train and val data frame
    :return dict with test, train and validation data frames
    """
    print("Loading test, train and validation data into pandas data-frames...")
    if read_head:
        return {
            "test_data_frame": pd.read_csv(paths["TEST_CSV"], nrows=5).drop("Unnamed: 0", axis=1),
            "train_data_frame": pd.read_csv(paths["TRAIN_CSV"], nrows=5).drop("Unnamed: 0", axis=1),
            "val_data_frame": pd.read_csv(paths["VAL_CSV"], nrows=5).drop("Unnamed: 0", axis=1),
        }
    return {
        "test_data_frame": pd.read_csv(paths["TEST_CSV"]).drop("Unnamed: 0", axis=1),
        "train_data_frame": pd.read_csv(paths["TRAIN_CSV"]).drop("Unnamed: 0", axis=1),
        "val_data_frame": pd.read_csv(paths["VAL_CSV"]).drop("Unnamed: 0", axis=1),
    }


def display_prepared_data(data_dict):
    """
    Display head of pandas data frames from given dictionary.
    :param data_dict: dict of train, test, validation pandas data frames
    """
    for key, data_frame in data_dict.items():
        src.helpers.print_extensions.print_variable(key)
        IPython.display.display(data_frame.head(5))


