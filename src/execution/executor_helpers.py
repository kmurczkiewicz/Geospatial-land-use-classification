import os

import tensorflow as tf


def test_saved_networks(paths, data):
    """
    Function to test all networks saved in default networks directory.
    :param paths: dict of app paths
    :param data: dict of test, train and validation data to be used in training
    """
    for network_name in os.listdir(paths["NETWORK_SAVE_DIR"]):
        print(f"Testing: {network_name}")
        tmp_network = tf.keras.models.load_model(
            os.path.join(paths["NETWORK_SAVE_DIR"], network_name)
        )
        test_loss, test_acc = tmp_network.evaluate(data["X_test"],  data["y_test"], verbose=1)
        print("\n")
