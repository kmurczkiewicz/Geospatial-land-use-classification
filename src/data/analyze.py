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
    src.helpers.print_extensions.print_subtitle("2. Data distribution ")
    
    data_distribution = {
        "train_data_frame": {key: 0 for key, _ in paths["LABEL_MAP"].items()},
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
        src.helpers.print_extensions.print_variable(key)
        display_values_distribution_values(distribution_values)
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
    x = [paths["LABEL_MAP"][key] for key in data_distribution[key].keys()]
    y = data_distribution[key].values()
    new_colors = ["red", "green", "blue", "yellow", "brown", "pink", "orange", "purple", "cyan"]
    matplotlib.pyplot.bar(x, y, color=new_colors)
    matplotlib.pyplot.xlabel('Class label')
    matplotlib.pyplot.ylabel('Amount in dataframe')
    matplotlib.pyplot.xticks(x)
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
    :param data_distribution_values: dict containing distribution details of test,
    train and validation data frames.
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


def analyze_saved_networks(paths):
    """
    Function to analyze saved networks in base app dir. For each network,
    details .json file is loaded into pandas data frame and displayed. Networks
    are sorted in data frame descending by FTA (First Test Accuracy).
    :param paths: dict of app paths
    """
    NN_DETAILS_FILE = "network_details.json"
    pandas_json_objects_list = []
    for network_name in os.listdir(paths["NETWORK_SAVE_DIR"]):
        pandas_json_object = pd.read_json(
            os.path.join(paths["NETWORK_SAVE_DIR"], network_name, NN_DETAILS_FILE),
            lines=True
        )
        pandas_json_objects_list.append(pandas_json_object)

    IPython.display.display(
        pd.concat(
            pandas_json_objects_list,
            ignore_index=True
        ).sort_values(['FTA'], ascending=[False])
    )