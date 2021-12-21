import numpy as np
import os
import PIL.Image


""" 3. Load train, test and validation data into memory """
def load_data_into_memory(paths, data_dict):


    test_data  = [np.asarray(PIL.Image.open(os.path.join(paths["DATASET"], data_dict["test_data_frame"]["Filename"][i]))) for i in range(len(data_dict["test_data_frame"]["Filename"]))]
    train_data = [np.asarray(PIL.Image.open(os.path.join(paths["DATASET"], data_dict["train_data_frame"]["Filename"][i]))) for i in range(len(data_dict["train_data_frame"]["Filename"]))]
    val_data   = [np.asarray(PIL.Image.open(os.path.join(paths["DATASET"], data_dict["val_data_frame"]["Filename"][i]))) for i in range(len(data_dict["val_data_frame"]["Filename"]))]

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