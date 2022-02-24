import datetime
import json
import os

import tensorflow as tf

from kerastuner.tuners import RandomSearch



class NetworkTuner:
    """
    Class to manage Network Tuner behavior.
    """

    def __init__(self, max_trials, executions_per_trial, n_epoch_search, hyper_model):
        self.max_trials = max_trials
        self.executions_per_trial = executions_per_trial
        self.hyper_model = hyper_model
        self.n_epoch_search = n_epoch_search

        self.tuner = None

    def initialize_tuner(self, overwrite):
        self.tuner = RandomSearch(
            self.hyper_model,
            objective='val_accuracy',
            seed=1,
            max_trials=self.max_trials,
            executions_per_trial=self.executions_per_trial,
            directory='random_search',
            project_name='geo_sat_nn_app',
            overwrite=overwrite
        )
        self.tuner.search_space_summary()

    def hyper_params_search(self, data):
        self.tuner.search(
            data["X_train"],
            data["y_train"],
            epochs=self.n_epoch_search,
            validation_data=(data["X_val"], data["y_val"]),
            callbacks=[tf.keras.callbacks.EarlyStopping('val_loss', patience=5)]
        )
        self.tuner.results_summary()

    def save_best_model(self, directory, data):
        best_model = self.tuner.get_best_models(num_models=1)[0]
        loss, accuracy = best_model.evaluate(data["X_test"], data["y_test"])

        date_time = datetime.datetime.now()
        model_name = f"CNN_HyperModel_{date_time.strftime('%H%M%d%m%y')}"
        model_save_dir = f"{directory}\\{model_name}"

        model_details = {
            "network_name" : model_name,
            "FTA"          : accuracy,
            "FTL"          : loss,
            "topology"     : "hyper_model",
            "optimizer"    : type(self.hyper_model.optimizer).__name__,
            "loss_function": self.hyper_model.loss_function,
            "created"      : date_time.strftime("%H:%M:%S, %d/%m/%Y"),
        }

        best_model.save(model_save_dir)

        # Save .json network descriptor
        with open(os.path.join(model_save_dir, "network_details.json"), 'w') as file:
            json.dump(model_details, file)

        # Save hyper parameters tuning logs
        with open(os.path.join(model_save_dir, "hyper_parameters_tuning.txt"), 'w') as file:
            # Adjust to network_hyper_model
            best_hyper_params = f"learning_rate : {self.tuner.get_best_hyperparameters()[0].get('learning_rate')}\n"\
                                f"batch_size : {self.tuner.get_best_hyperparameters()[0].get('batch_size')}\n"\
                                f"dropout : {self.tuner.get_best_hyperparameters()[0].get('dropout')}\n"
            file.write(best_hyper_params)
