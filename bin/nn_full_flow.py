import tensorflow as tf

import src.execution.main_executor

def main():
    executor = src.execution.main_executor.MainExecutor(display=True)
    executor.execute_full_flow(
        topology     = "D",
        epochs       = 10,
        optimizer    = tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss_function= tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics      = ['accuracy'],
        save_model   = True
    )

if __name__ == "__main__" :
        main()
