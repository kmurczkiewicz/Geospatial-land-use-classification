import tensorflow as tf

import src.execution.main_executor

OPTIMIZERS = [
  tf.keras.optimizers.Adadelta,
  tf.keras.optimizers.Adagrad,
  tf.keras.optimizers.Adam,
  tf.keras.optimizers.Adamax,
  tf.keras.optimizers.Ftrl,
  tf.keras.optimizers.Nadam,
  tf.keras.optimizers.RMSprop,
  tf.keras.optimizers.SGD
]

ACTIVATIONS = ['relu', 'tanh', 'sigmoid']

def main():
    for optimizer in OPTIMIZERS:
        for activation in ACTIVATIONS:
            executor = src.execution.main_executor.MainExecutor(display=True)
            executor.execute_full_flow(
                topology     = "D",
                epochs       = 100,
                optimizer    = optimizer(),
                loss_function= tf.keras.losses.SparseCategoricalCrossentropy(),
                batch_size   = 128,
                metrics      = ['accuracy'],
                layer_activation_function=activation,
                save_model   = True
            )

if __name__ == "__main__" :
        main()