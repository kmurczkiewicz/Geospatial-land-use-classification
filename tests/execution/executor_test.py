import os
import pytest
import shutil

import tensorflow as tf

import src.execution.main_executor
import src.nn_library.network

EXAMPLE_NETWORK = "./artefacts/example_network_model"



def test_init_executor():
    """
    Function to test if init_executor is executed correctly and no exception is raised.
    """
    try:
        _ = src.execution.main_executor.MainExecutor(display=False)
    except Exception:
        pytest.fail("Failed stage: init_executor")


def test_stage_prepare_data():
    """
    Function to test if stage_prepare_data is executed correctly and no exception is raised.
    """
    try:
        executor = src.execution.main_executor.MainExecutor(display=False)
        executor.stage_prepare_data(read_head=True)
    except Exception:
        pytest.fail("Failed stage: stage_prepare_data")


def test_stage_analyze_data():
    """
    Function to test if stage_analyze_data is executed correctly and no exception is raised.
    """
    try:
        executor = src.execution.main_executor.MainExecutor(display=False)
        executor.stage_analyze_data(
            executor.stage_prepare_data(read_head=True),
            display=False
        )
    except Exception:
        pytest.fail("Failed stage: stage_analyze_data")


def test_stage_load_data():
    """
    Function to test if stage_load_data is executed correctly and no exception is raised.
    """
    try:
        executor = src.execution.main_executor.MainExecutor(display=False)
        _ = executor.stage_load_data(
            executor.stage_prepare_data(read_head=True)
        )
    except Exception:
        pytest.fail("Failed stage: stage_load_data")


def test_stage_test_saved_networks():
    """
    Function to test if stage_test_saved_networks is executed correctly and no exception is raised.
    """
    try:
        executor = src.execution.main_executor.MainExecutor(display=False)
        executor.stage_test_saved_networks(
            executor.stage_load_data(
                executor.stage_prepare_data(read_head=True)
            )
        )
    except Exception:
        pytest.fail("Failed stage: stage_test_saved_networks")


def test_stage_analyze_saved_networks():
    """
    Function to test if stage_analyze_saved_networks is executed correctly and no exception is raised.
    """
    try:
        executor = src.execution.main_executor.MainExecutor(display=False)
        executor.stage_analyze_saved_networks()
    except Exception:
        pytest.fail("Failed stage: stage_analyze_saved_networks")


def test_stage_nn_init():
    """
    Function to test if stage_nn_init is executed correctly and no exception is raised.
    """
    try:
        executor = src.execution.main_executor.MainExecutor(display=False)
        _ = executor.stage_nn_init(
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
        executor = src.execution.main_executor.MainExecutor(display=False)
        tmp_network = src.nn_library.network.Neural_network()
        tmp_network.model = tf.keras.models.load_model(EXAMPLE_NETWORK)
        executor.stage_nn_train(
            tmp_network,
            executor.stage_load_data(
                executor.stage_prepare_data(read_head=True)
            ),
            1
        )
    except Exception:
        pytest.fail("Failed stage: stage_nn_train")


def test_stage_nn_test():
    """
    Function to test if stage_nn_test is executed correctly and no exception is raised.
    """
    try:
        executor = src.execution.main_executor.MainExecutor(display=False)
        tmp_network = src.nn_library.network.Neural_network()
        tmp_network.model = tf.keras.models.load_model(EXAMPLE_NETWORK)
        executor.stage_nn_test(
            tmp_network,
            executor.stage_load_data(
                executor.stage_prepare_data(read_head=True)
            )
        )
    except Exception:
        pytest.fail("Failed stage: stage_nn_train")


def test_stage_nn_save():
    """
    Function to test if stage_nn_save is executed correctly and no exception is raised.
    tf.keras.models.load_model(EXAMPLE_NETWORK)
    """
    try:
        TEST_MODEL_NAME = 'TEST_MODEL'
        executor = src.execution.main_executor.MainExecutor(display=False)
        tmp_network = src.nn_library.network.Neural_network()
        tmp_network.model = tf.keras.models.load_model(EXAMPLE_NETWORK)
        executor.stage_nn_save(
            executor.PATHS["NETWORK_SAVE_DIR"],
            TEST_MODEL_NAME,
            tmp_network
        )
        # Remove network created during testing
        for network_name in os.listdir(executor.PATHS["NETWORK_SAVE_DIR"]):
            if TEST_MODEL_NAME in network_name:
                shutil.rmtree(os.path.join(executor.PATHS["NETWORK_SAVE_DIR"], network_name))
    except Exception:
        pytest.fail("Failed stage: stage_nn_save")