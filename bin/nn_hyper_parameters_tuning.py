import tensorflow as tf

import src.execution.main_executor

def main():
    executor = src.execution.main_executor.MainExecutor(display=True)
    executor.execute_nn_hyper_parameters_tuning(
        overwrite=True,
        max_trials=1,
        executions_per_trial=1,
        n_epoch_search=1,
        save_model=True
    )

if __name__ == "__main__" :
        main()