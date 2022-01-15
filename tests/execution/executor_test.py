import os
import pytest
import shutil

import tensorflow as tf

import bin.execution.executor
import src.execution.executor_source


def test_init_executor():
    """
    Function to test if init_executor is executed correctly and no exception is raised.
    """
    try:
        _ = src.execution.executor_source.init_executor()
    except Exception:
        pytest.fail("Failed stage: init_executor")


def test_stage_prepare_data():
    """
    Function to test if stage_prepare_data is executed correctly and no exception is raised.
    """
    try:
        executor = bin.execution.executor.Executor(display=False)
        src.execution.executor_source.stage_prepare_data(executor.PATHS, read_head=True)
    except Exception:
        pytest.fail("Failed stage: stage_prepare_data")


def test_stage_analyze_data():
    """
    Function to test if stage_analyze_data is executed correctly and no exception is raised.
    """
    try:
        executor = bin.execution.executor.Executor(display=False)
        data_dict = src.execution.executor_source.stage_prepare_data(executor.PATHS, read_head=True)
        src.execution.executor_source.stage_analyze_data(executor.PATHS, data_dict, display=False)
    except Exception:
        pytest.fail("Failed stage: stage_analyze_data")


def test_stage_load_data():
    """
    Function to test if stage_load_data is executed correctly and no exception is raised.
    """
    try:
        executor = bin.execution.executor.Executor(display=False)
        data_dict = src.execution.executor_source.stage_prepare_data(executor.PATHS, read_head=True)
        data = src.execution.executor_source.stage_load_data(executor.PATHS, data_dict)
    except Exception:
        pytest.fail("Failed stage: stage_load_data")


def test_stage_test_saved_networks():
    """
    Function to test if stage_test_saved_networks is executed correctly and no exception is raised.
    """
    try:
        executor = bin.execution.executor.Executor(display=False)
        data_dict = src.execution.executor_source.stage_prepare_data(executor.PATHS, read_head=True)
        data = src.execution.executor_source.stage_load_data(executor.PATHS, data_dict)
        src.execution.executor_source.stage_test_saved_networks(executor.PATHS, data)
    except Exception:
        pytest.fail("Failed stage: stage_test_saved_networks")


def test_stage_analyze_saved_networks():
    """
    Function to test if stage_analyze_saved_networks is executed correctly and no exception is raised.
    """
    try:
        executor = bin.execution.executor.Executor(display=False)
        src.execution.executor_source.stage_analyze_saved_networks(executor.PATHS)
    except Exception:
        pytest.fail("Failed stage: stage_analyze_saved_networks")


def test_stage_nn_init():
    """
    Function to test if stage_nn_init is executed correctly and no exception is raised.
    """
    try:
        executor = bin.execution.executor.Executor(display=False)
        _ = src.execution.executor_source.stage_nn_init(
            nn_topology=executor.NN_TOPOLOGIES["TEST"],
            input_shape=(64, 64, 3),
            optimizer=tf.keras.optimizers.Adam(),
            loss_function=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )
    except Exception:
        pytest.fail("Failed stage: stage_nn_init")


def test_stage_nn_train():
    """
    Function to test if stage_nn_train is executed correctly and no exception is raised.
    """
    try:
        executor = bin.execution.executor.Executor(display=False)
        data_dict = src.execution.executor_source.stage_prepare_data(executor.PATHS, read_head=True)
        data = src.execution.executor_source.stage_load_data(executor.PATHS, data_dict)
        model = src.execution.executor_source.stage_nn_init(
            nn_topology=executor.NN_TOPOLOGIES["TEST"],
            input_shape=(64, 64, 3),
            optimizer=tf.keras.optimizers.Adam(),
            loss_function=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )
        src.execution.executor_source.stage_nn_train(model, data, 1)
    except Exception:
        pytest.fail("Failed stage: stage_nn_train")


def test_stage_nn_test():
    """
    Function to test if stage_nn_test is executed correctly and no exception is raised.
    """
    try:
        executor = bin.execution.executor.Executor(display=False)
        data_dict = src.execution.executor_source.stage_prepare_data(executor.PATHS, read_head=True)
        data = src.execution.executor_source.stage_load_data(executor.PATHS, data_dict)
        model = src.execution.executor_source.stage_nn_init(
            nn_topology=executor.NN_TOPOLOGIES["TEST"],
            input_shape=(64, 64, 3),
            optimizer=tf.keras.optimizers.Adam(),
            loss_function=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )
        src.execution.executor_source.stage_nn_train(model, data, 1)
        src.execution.executor_source.stage_nn_test(model, data)
    except Exception:
        pytest.fail("Failed stage: stage_nn_train")


def test_stage_nn_save():
    """
    Function to test if stage_nn_save is executed correctly and no exception is raised.
    """
    try:
        TEST_MODEL_NAME = 'TEST_MODEL'
        executor = bin.execution.executor.Executor(display=False)
        data_dict = src.execution.executor_source.stage_prepare_data(executor.PATHS, read_head=True)
        data = src.execution.executor_source.stage_load_data(executor.PATHS, data_dict)
        model = src.execution.executor_source.stage_nn_init(
            nn_topology=executor.NN_TOPOLOGIES["TEST"],
            input_shape=(64, 64, 3),
            optimizer=tf.keras.optimizers.Adam(),
            loss_function=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )
        src.execution.executor_source.stage_nn_train(model, data, 1)
        src.execution.executor_source.stage_nn_test(model, data)
        src.execution.executor_source.stage_nn_save(
            executor.PATHS["NETWORK_SAVE_DIR"],
            TEST_MODEL_NAME,
            model
        )
        # Remove network created during testing
        for network_name in os.listdir(executor.PATHS["NETWORK_SAVE_DIR"]):
            if TEST_MODEL_NAME in network_name:
                shutil.rmtree(os.path.join(executor.PATHS["NETWORK_SAVE_DIR"], network_name))
    except Exception:
        pytest.fail("Failed stage: stage_nn_train")