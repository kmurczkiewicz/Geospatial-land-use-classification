import re


def natural_sort(list_to_sort):
    """
    Function to perform natural sort on given list.
    :param list_to_sort: list to be sorted using natural sort algorithm
    :return: list sorted in natural sort order
    """
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(list_to_sort, key=alphanum_key)


def get_classification_color(class_name):

    CLASS_COLOR = {
        "AnnualCrop"           : "#FFAE00",
        "Forest"               : "#1EFF1E",
        "HerbaceousVegetation" : "#8353DC",
        "Highway"              : "#FF1EF1",
        "Industrial"           : "#F50318",
        "Pasture"              : "#A2F91D",
        "PermanentCrop"        : "#DFD433",
        "Residential"          : "#98A0A2",
        "River"                : "#06BDF8",
        "SeaLake"              : "#0648F8",
    }

    return CLASS_COLOR[class_name]
