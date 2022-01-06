import tensorflow as tf

import src.helpers.timer
import src.helpers.print_extensions

import src.data.analyze
import src.data.load
import src.data.prepare

import src.nn_library.network


def stage_prepare_data(paths):
    """
    Execute data preparation stage.
    Load test, train and validation .csv files into pandas data frames.
    :param paths: dict of app paths
    :return dict of train, test, validation pandas data frames
    """
    timer = src.helpers.timer.Timer()
    src.helpers.print_extensions.print_title("1. Prepare test, train and validation data")
    timer.set_timer()
    data_dict = src.data.prepare.data_init(paths)
    timer.stop_timer()
    src.helpers.print_extensions.print_border()
    return data_dict


def stage_analyze_data(paths, data_dict):
    """
    Execute analyze data stage.
    Display label map, amount of class labels in test, train and validation data frame,
    and plot 5x5 images from each dataframe.
    :param paths: dict of app paths
    :param data_dict: dict of train, test, validation pandas data frames
    """
    timer = src.helpers.timer.Timer()
    timer.set_timer()
    src.helpers.print_extensions.print_title("2. Analyze test, train and validation data")
    src.data.analyze.analyze_data(paths, data_dict)
    timer.stop_timer()


def stage_load_data(paths, data_dict):
    """
    Execute load data stage.
    Load test, train and validation data into memory.
    :param paths: dict of app paths
    :param data_dict: dict of train, test, validation pandas data frames
    :return: dict of test, train and validation data
    """
    timer = src.helpers.timer.Timer()
    timer.set_timer()
    src.helpers.print_extensions.print_title("2. Load test, train and validation data into memory")
    data = src.data.load.load_into_memory(paths, data_dict)
    timer.stop_timer()
    src.helpers.print_extensions.print_border()
    return data


def stage_nn_init(nn_topology, input_shape):
    """
    Execute neural network initialization stage.
    Create network object and compile the network.
    :param nn_topology: str topology name to be used
    :param input_shape: tuple of three integers
    :return: initialized network model
    """
    timer = src.helpers.timer.Timer()
    timer.set_timer()
    src.helpers.print_extensions.print_title("3. Create and compile CNN model")

    cnn_model = src.nn_library.network.Neural_network(nn_topology, input_shape)
    cnn_model.init_network()
    cnn_model.compile(
        'adam',
        tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        ['accuracy']
    )
    timer.stop_timer()
    src.helpers.print_extensions.print_border()
    return cnn_model


def stage_nn_train(cnn_model: src.nn_library.network.Neural_network, data):
    """
    Execute network training stage.
    :param cnn_model: object of type src.nn_library.network.Neural_network
    :param data: dict of test, train and validation data to be used in training
    """
    timer = src.helpers.timer.Timer()
    src.helpers.print_extensions.print_title("4. Train CNN model")
    timer.set_timer()
    cnn_model.train_cnn_model(data, epochs_num=10)
    timer.stop_timer()
    cnn_model.plot_model_result()


def stage_nn_test(cnn_model: src.nn_library.network.Neural_network, data):
    """
    Execute network testing stage.
    :param cnn_model: object of type src.nn_library.network.Neural_network
    :param data: dict of test, train and validation data to be used in training
    """
    src.helpers.print_extensions.print_title("5. Test the CNN model")
    cnn_model.test_network(data)


def stage_nn_save(save_dir, network_name, cnn_model: src.nn_library.network.Neural_network):
    """
    Execute save network stage. Save network in given dir with given name in .pb format.
    :param save_dir: str where network model will be saved in .pb format
    :param network_name: str name of saved network
    :param cnn_model: object of type src.nn_library.network.Neural_network
    """
    src.helpers.print_extensions.print_title("6. Save the model")
    cnn_model.save_model(network_name, save_dir)
