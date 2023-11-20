import optuna
import logging
import sys

# Add stream handler of stdout to show the messages
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
study_name = "example-study"  # Unique identifier of the study.
storage_name = "sqlite:///{}.db".format(study_name)
study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True)

def objective(trial):
    x = trial.suggest_float("x", -10, 10)
    return (x - 2) ** 2

study.optimize(objective, n_trials=3)
print(f"Best value: {study.best_value} (params: {study.best_params})")
