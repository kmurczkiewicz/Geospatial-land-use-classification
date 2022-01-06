import os

import PIL.Image
import matplotlib.pyplot

import src.helpers.print_extensions


def analyze_data(paths, data_dict):
    """
    Function to analyze data distribution in each of train, test and val dataframe.
    :param paths: dict of app paths
    :param data_dict: dict of train, test, validation pandas data frames
    """
    src.helpers.print_extensions.print_variable("LABEL_MAP:")
    src.helpers.print_extensions.print_dict(paths["LABEL_MAP"])
    
    src.helpers.print_extensions.print_subtitle("2.1. Data distribution ")
    
    data_distribution = {
        "test_data_frame" : {key : 0 for key, _ in paths["LABEL_MAP"].items()},
        "train_data_frame": {key : 0 for key, _ in paths["LABEL_MAP"].items()},
        "val_data_frame"  : {key : 0 for key, _ in paths["LABEL_MAP"].items()}
    }

    for key, value_dict in data_distribution.items():
        for sub_key, _ in value_dict.items():
            details = data_dict[key].apply(
                lambda x: True if x['Label'] == paths["LABEL_MAP"][sub_key] else False, axis=1
            )
            # DO NOT CHANGE THIS CONDITION: details == True TO: details is True, it will break the logic
            value_dict[sub_key] = len(details[details == True].index)
        plot(paths, data_distribution, key)
        show_example(paths, data_dict, key)


def plot(paths, data_distribution, key):
    """
    Function to plot data distribution in dictionary.
    :param paths: dict of app paths
    :param data_distribution: dict of data distribution in test, train and val data frames
    :param key: str key to data_distribution
    """
    src.helpers.print_extensions.print_variable(key)
    x = [paths["LABEL_MAP"][key] for key in data_distribution[key].keys()]
    y = data_distribution[key].values()
    new_colors = ["red", "green", "blue", "yellow", "brown", "pink", "orange", "purple", "cyan"]
    matplotlib.pyplot.bar(x, y, color=new_colors)
    matplotlib.pyplot.xlabel('Class label')
    matplotlib.pyplot.ylabel('Amount in dataframe')
    matplotlib.pyplot.xticks(x)
    matplotlib.pyplot.show()
    

def show_example(paths, data_dict, key):
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