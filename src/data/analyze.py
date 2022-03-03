import os

import PIL.Image
import matplotlib.pyplot
import IPython.display

import pandas as pd

import src.helpers


def analyze_data(paths, data_dict, display):
    """
    Function to analyze data distribution in each of train, test and val dataframe.

    :param paths: dict of app paths
    :param data_dict: dict of train, test, validation pandas data frames
    :param display: bool type, display output (images, plots etc.)
    """
    src.helpers.print_extensions.print_variable("LABEL_MAP:")
    src.helpers.print_extensions.print_dict(paths["LABEL_MAP"])
    print("\n\n")
    src.helpers.print_extensions.print_subtitle("Data distribution ")
    
    data_distribution = {
        "train_data_frame": {key : 0 for key, _ in paths["LABEL_MAP"].items()},
        "test_data_frame" : {key : 0 for key, _ in paths["LABEL_MAP"].items()},
        "val_data_frame"  : {key : 0 for key, _ in paths["LABEL_MAP"].items()}
    }

    distribution_values = {}

    for key, value_dict in data_distribution.items():
        for sub_key, _ in value_dict.items():
            details = data_dict[key].apply(
                lambda x: True if x['Label'] == paths["LABEL_MAP"][sub_key] else False, axis=1
            )
            # DO NOT CHANGE THIS CONDITION: details == True TO: details is True, it will break the logic
            value_dict[sub_key] = len(details[details == True].index)
            distribution_values[sub_key] = {
                "amount" : len(details[details == True].index),
                "label"  : paths["LABEL_MAP"][sub_key],
                "data_frame": key
            }
        if not display:
            continue

        sum_amount = sum(dict_value["amount"] for dict_value in distribution_values.values())
        src.helpers.print_extensions.print_variable(str(key + " - " + str(sum_amount) + " images"))
        for class_name, dict_value in distribution_values.items():
            percent_of_all = "{:.2f}".format((dict_value['amount'] / sum_amount) * 100)
            print(f"{class_name} - {percent_of_all}%")
        # display_values_distribution_values(distribution_values)
        plot(paths, data_distribution, key)
        display_example(paths, data_dict, key)
        print("\n")
        src.helpers.print_extensions.print_border()


def plot(paths, data_distribution, key):
    """
    Function to plot data distribution in dictionary.

    :param paths: dict of app paths
    :param data_distribution: dict of data distribution in test, train and val data frames
    :param key: str key to data_distribution
    """
    x = [f"{key} [{paths['LABEL_MAP'][key]}]" for key in data_distribution[key].keys()]
    y = data_distribution[key].values()
    new_colors = ["red", "yellow", "orange", "green", "cyan", "blue", "purple", "magenta", "lime", "brown"]
    matplotlib.pyplot.figure(figsize=(10, 4.8))
    matplotlib.pyplot.barh(x, y, color=new_colors)
    matplotlib.pyplot.xlabel(f'Amount in {key}')
    matplotlib.pyplot.ylabel('Class')
    matplotlib.pyplot.yticks(x)
    # matplotlib.pyplot.xticks(x)
    for index, value in enumerate(y):
        matplotlib.pyplot.text(value, index, str(value))
    matplotlib.pyplot.show()


def display_example(paths, data_dict, key):
    """
    Function to display 5x5 grid with images and their labels.

    :param paths: dict of app paths
    :param data_dict: dict of train, test, validation pandas data frames
    :param key: str key to data_dict
    """
    matplotlib.pyplot.figure(figsize=(10, 10))
    for i in range(25):
        matplotlib.pyplot.subplot(5, 5, i+1)
        matplotlib.pyplot.xticks([])
        matplotlib.pyplot.yticks([])
        matplotlib.pyplot.grid(False)
        matplotlib.pyplot.imshow(PIL.Image.open(os.path.join(paths["DATASET"], data_dict[key]["Filename"][i])))
        matplotlib.pyplot.xlabel(
            data_dict[key]["ClassName"][i] + f" ({paths['LABEL_MAP'][data_dict[key]['ClassName'][i]]})"
        )
    matplotlib.pyplot.show()


def display_values_distribution_values(data_distribution_values):
    """
    Function to display quantitative distribution of classes
    in test, train and validation data frames.

    :param data_distribution_values: dict containing distribution details of test, train and validation data frames.
    """
    sum_amount = 0
    for key, dict_value in data_distribution_values.items():
        sum_amount = sum_amount + dict_value["amount"]

    for key, dict_value in data_distribution_values.items():
        percent_of_all = "{:.2f}".format((dict_value['amount'] / sum_amount) * 100)
        print(
            f"{key}:"
            f"\n\t- class label: {dict_value['label']}"
            f"\n\t- amount: {dict_value['amount']}"
            f"\n\t- {percent_of_all}% of {dict_value['data_frame']}"
        )

def _get_network_details(network_path):
    return pd.read_json(
        network_path,
        lines=True
    )

def analyze_saved_networks(paths, nn_dir):
    """
    Function to analyze saved networks in base app dir. For each network,
    details .json file is loaded into pandas data frame and displayed. Networks
    are sorted in data frame descending by FTA (First Test Accuracy).

    :param paths: dict of app paths
    :param nn_dir: str name of directory from which network shall be analyzed, if empty,
        recursively search for saved networks and analyze them all.
    """
    NN_DETAILS_FILE = "network_details.json"
    pandas_json_objects_list = []

    if not nn_dir:
        # Analyze all networks recursively
        for root, dirs, files in os.walk(paths["NETWORK_SAVE_DIR"]):
            for name in files:
                if name == "saved_model.pb":
                    pandas_json_objects_list.append(_get_network_details(os.path.join(root, NN_DETAILS_FILE)))
    else:
        # Analyze networks in given directory
        for root, dirs, files in os.walk(os.path.join(paths["NETWORK_SAVE_DIR"], nn_dir)):
            for name in files:
                if name == "saved_model.pb":
                    pandas_json_objects_list.append(_get_network_details(os.path.join(root, NN_DETAILS_FILE)))

    IPython.display.display(
        pd.concat(
            pandas_json_objects_list,
            ignore_index=True
        ).sort_values(['FTA'], ascending=[False])
    )
