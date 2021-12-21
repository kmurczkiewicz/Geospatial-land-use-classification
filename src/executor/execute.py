import src.helpers.timer
import src.helpers.print_extensions

import src.data.analyze
import src.data.load
import src.data.prepare

import src.nn_operations.basic

class Path_holder():
    PATHS = {}


def stage_set_paths(paths):
    Path_holder.PATHS = paths

def stage_preparation():
    timer = src.helpers.timer.Timer()
    src.helpers.print_extensions.print_title("1. Prepare test, train and validation data")
    timer.set_timer()
    data_dict = src.data.prepare.data_init(Path_holder.PATHS)
    timer.stop_timer()
    src.helpers.print_extensions.print_border()
    return data_dict


def stage_analyze_data(data_dict):
    timer = src.helpers.timer.Timer()
    timer.set_timer()
    src.helpers.print_extensions.print_title("2. Analyze test, train and validation data")
    src.data.analyze.analyze_data(Path_holder.PATHS, data_dict)
    timer.stop_timer()
    return None

def stage_load_data(data_dict):
    timer = src.helpers.timer.Timer()
    timer.set_timer()
    src.helpers.print_extensions.print_title("2. Load test, train and validation data into memory")
    data = src.data.load.load_into_memory(Path_holder.PATHS, data_dict)
    timer.stop_timer()
    src.helpers.print_extensions.print_border()
    return data


def stage_nn_init():
    timer = src.helpers.timer.Timer()
    timer.set_timer()
    src.helpers.print_extensions.print_title("3. Create and compile CNN model")
    cnn_model = src.nn_operations.basic.create_cnn_model()
    src.nn_operations.basic.compile_cnn_model(cnn_model)
    timer.stop_timer()
    src.helpers.print_extensions.print_border()
    return cnn_model


def stage_nn_train(cnn_model, data):
    timer = src.helpers.timer.Timer()
    src.helpers.print_extensions.print_title("4. Train CNN model")
    timer.set_timer()
    training_history = src.nn_operations.basic.train_cnn_model(cnn_model, data)
    timer.stop_timer()
    src.nn_operations.basic.plot_model_result(training_history)
    return training_history


def stage_nn_test(cnn_model, data):
    src.helpers.print_extensions.print_title("5. Test the CNN model")
    src.nn_operations.basic.test_cnn_model(cnn_model, data)


def stage_nn_save(cnn_model):
    src.helpers.print_extensions.print_title("6. Save the model")
    src.nn_operations.basic.save_model(
        cnn_model,
        "small_cnn",
        "models_pb"
    )
