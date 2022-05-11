import tensorflow as tf

import src.execution.main_executor

OPTIMIZERS = [
    #tf.keras.optimizers.Adagrad,
    tf.keras.optimizers.Adam,
    #tf.keras.optimizers.Adamax,
    #tf.keras.optimizers.RMSprop,
    #tf.keras.optimizers.SGD
]

ACTIVATIONS = ['relu', 'tanh', 'selu', 'elu']

def main():
    for optimizer in OPTIMIZERS:
        for activation in ACTIVATIONS:
            executor = src.execution.main_executor.MainExecutor(display=True)
            executor.execute_full_flow(
                architecture = "D",
                epochs       = 100,
                optimizer    = optimizer(learning_rate=0.00046748),
                loss_function= tf.keras.losses.SparseCategoricalCrossentropy(),
                batch_size   = 128,
                metrics      = ['accuracy'],
                layer_activation_function=activation,
                save_model   = True
            )

if __name__ == "__main__" :
        main()
