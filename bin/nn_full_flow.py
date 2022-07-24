import tensorflow as tf
import src.execution.main_executor
from tensorflow.python.client import device_lib


def main():
    # Check if GPU is available
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    device_lib.list_local_devices()

    OPTIMIZERS = [
        {"optimizer": tf.keras.optimizers.Adagrad},
        {"optimizer": tf.keras.optimizers.Adam, "learning_rate": 0.00046748},
        {"optimizer": tf.keras.optimizers.Adamax},
        {"optimizer": tf.keras.optimizers.RMSprop},
        {"optimizer": tf.keras.optimizers.SGD, "learning_rate": 0.0075}
    ]

    ACTIVATIONS = ['relu', 'tanh', 'selu', 'elu']

    executor = src.execution.main_executor.MainExecutor(display=True)
    executor.execute_full_flow(
        architecture="D",
        epochs=100,
        optimizers=OPTIMIZERS,
        loss_function=tf.keras.losses.SparseCategoricalCrossentropy(),
        batch_size=128,
        metrics=['accuracy'],
        activations=ACTIVATIONS,
        save_model=False
    )

if __name__ == "__main__" :
        main()
