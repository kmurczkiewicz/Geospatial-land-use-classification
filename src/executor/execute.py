import tensorflow as tf

import src.helpers.timer
import src.helpers.print_extensions

import src.data.analyze
import src.data.load
import src.data.prepare

import src.nn_library.network


def stage_preparation(paths):
    timer = src.helpers.timer.Timer()
    src.helpers.print_extensions.print_title("1. Prepare test, train and validation data")
    timer.set_timer()
    data_dict = src.data.prepare.data_init(paths)
    timer.stop_timer()
    src.helpers.print_extensions.print_border()
    return data_dict


def stage_analyze_data(paths, data_dict):
    timer = src.helpers.timer.Timer()
    timer.set_timer()
    src.helpers.print_extensions.print_title("2. Analyze test, train and validation data")
    src.data.analyze.analyze_data(paths, data_dict)
    timer.stop_timer()


def stage_load_data(paths, data_dict):
    timer = src.helpers.timer.Timer()
    timer.set_timer()
    src.helpers.print_extensions.print_title("2. Load test, train and validation data into memory")
    data = src.data.load.load_into_memory(paths, data_dict)
    timer.stop_timer()
    src.helpers.print_extensions.print_border()
    return data


def stage_nn_init(input_shape):
    timer = src.helpers.timer.Timer()
    timer.set_timer()
    src.helpers.print_extensions.print_title("3. Create and compile CNN model")

    cnn_model = src.nn_library.network.Neural_network(input_shape)
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
    timer = src.helpers.timer.Timer()
    src.helpers.print_extensions.print_title("4. Train CNN model")
    timer.set_timer()
    cnn_model.train_cnn_model(data, epochs_num=10)
    timer.stop_timer()
    cnn_model.plot_model_result()


def stage_nn_test(cnn_model: src.nn_library.network.Neural_network, data):
    src.helpers.print_extensions.print_title("5. Test the CNN model")
    cnn_model.test_network(data)


def stage_nn_save(save_dir, network_name, cnn_model: src.nn_library.network.Neural_network):
    src.helpers.print_extensions.print_title("6. Save the model")
    cnn_model.save_model(network_name, save_dir)
