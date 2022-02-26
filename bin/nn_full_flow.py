import tensorflow as tf

import src.execution.main_executor

def main():
    executor = src.execution.main_executor.MainExecutor(display=True)
    executor.execute_full_flow(
        topology     = "D",
        epochs       = 5,
        optimizer    = tf.keras.optimizers.Adam(learning_rate=3e-4),
        loss_function= tf.keras.losses.SparseCategoricalCrossentropy(),
        batch_size   = 128,
        metrics      = ['accuracy'],
        layer_activation_function='relu',
        save_model   = False
    )

if __name__ == "__main__" :
        main()