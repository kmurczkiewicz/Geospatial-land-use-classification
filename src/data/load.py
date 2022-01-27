import numpy as np
import os
import PIL.Image


def load_into_memory(paths, data_dict):
    """
    Function to load train, test and validation data into memory
    :param paths: dict of app paths
    :param data_dict: dict of train, test, validation pandas data frames
    :return dict of images in type array and labels in test, train and validation form
    """
    print("Loading test, train and validation data into memory...")
    test_data  = [
        np.asarray(
            PIL.Image.open(
                os.path.join(paths["DATASET"], data_dict["test_data_frame"]["Filename"][i])
            )) for i in range(len(data_dict["test_data_frame"]["Filename"]))
    ]

    train_data = [
        np.asarray(
            PIL.Image.open(
                os.path.join(paths["DATASET"], data_dict["train_data_frame"]["Filename"][i])
            )) for i in range(len(data_dict["train_data_frame"]["Filename"]))
    ]

    val_data   = [
        np.asarray(
            PIL.Image.open(
                os.path.join(paths["DATASET"], data_dict["val_data_frame"]["Filename"][i])
            )) for i in range(len(data_dict["val_data_frame"]["Filename"]))
    ]

    # Normalize pixel values to be between 0 and 1
    test_data = [x / 255.0 for x in test_data]
    train_data = [x / 255.0 for x in train_data]
    val_data = [x / 255.0 for x in val_data]

    test_labels = [data_dict["test_data_frame"]["Label"][i] for i in range(len(data_dict["test_data_frame"]["Label"]))]
    train_labels= [data_dict["train_data_frame"]["Label"][i] for i in range(len(data_dict["train_data_frame"]["Label"]))]
    val_labels  = [data_dict["val_data_frame"]["Label"][i] for i in range(len(data_dict["val_data_frame"]["Label"]))]

    return {
        "X_train" : np.array(train_data),
        "y_train" : np.array(train_labels),
        "X_test"  : np.array(test_data),
        "y_test"  : np.array(test_labels),
        "X_val"   : np.array(val_data),
        "y_val"   : np.array(val_labels)
    }


def load_sat_image_as_array(paths, img_name):
    """
    Function to load a satellite image and resize its width and height to multiplication of 64.
    :param paths: dict of app paths
    :param img_name: str name of image to be loaded
    :return: image in numpy array type
    """
    img = PIL.Image.open(os.path.join(paths["SAT_IMG_PATH"], img_name))
    original_width, original_height = img.size
    fixed_width = original_width - (original_width % 64)
    fixed_height = original_height - (original_height % 64)
    img = img.resize((fixed_width, fixed_height), PIL.Image.ANTIALIAS)

    return np.asarray(img)

