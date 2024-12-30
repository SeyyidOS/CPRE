from torch.utils.tensorboard.writer import SummaryWriter
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import lightgbm as lgb
import optuna 


class LightGBMLearner:
    def __init__(self, params=None, log_dir="../runs/lgbm"):
        self.params = params if params else {
            'objective': 'regression',
            'metric': 'mse',
            'boosting_type': 'gbdt',
            'learning_rate': 0.1,
            'num_leaves': 31,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 1
        }
        self.model = None
        self.log_dir = log_dir

    def train(self, X_train, y_train, X_val, y_val):
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        writer = SummaryWriter(self.log_dir)

        def log_callback(env):
            for i, (_, name, metric, _) in enumerate(env.evaluation_result_list):
                writer.add_scalar(f"LightGBM/{name}", metric, env.iteration)

        self.model = lgb.train(
            self.params,
            train_data,
            valid_sets=[train_data, val_data],
            num_boost_round=500,
            callbacks=[log_callback],
        )

    def predict(self, X):
        if self.model is None:
            raise ValueError("The model has not been trained yet.")
        return self.model.predict(X, num_iteration=self.model.best_iteration)

    def evaluate(self, X_val, y_val):
        predictions = self.predict(X_val)
        mse = mean_squared_error(y_val, predictions, squared=True)
        print(f"Validation MSE: {mse:.4f}")
        return mse

    def load_model(self, model_path):
        self.model = lgb.Booster(model_file=model_path)

    def save_model(self, model_path):
        if self.model is None:
            raise ValueError("The model has not been trained yet.")
        self.model.save_model(model_path)

    def optimize(self, X_train, y_train, X_val, y_val, n_trials=50, db_path="sqlite:///lightgbm_optuna.db"):
        def objective(trial):
            params = {
                'objective': 'regression',
                'metric': 'mse',
                'boosting_type': 'gbdt',
                'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.1),
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'feature_fraction': trial.suggest_uniform('feature_fraction', 0.6, 1.0),
                'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.6, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
                'verbose': -1
            }

            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

            model = lgb.train(
                params,
                train_data,
                valid_sets=[val_data],
            )

            predictions = model.predict(X_val, num_iteration=model.best_iteration)
            mse = mean_squared_error(y_val, predictions, squared=True)
            return mse

        study = optuna.create_study(direction="minimize", study_name="lightgbm_optimization", storage=db_path, load_if_exists=True)
        study.optimize(objective, n_trials=n_trials)

        print("Best hyperparameters:", study.best_params)
        print("Best MSE:", study.best_value)

        self.params.update(study.best_params)
        return study.best_params


    def plot_predictions_histogram(self, y_true, y_pred, bins=30, title="Ground Truth vs Predictions Histogram"):
        plt.figure(figsize=(10, 6))
        plt.hist(y_true, bins=bins, alpha=0.5, label="Ground Truth", color="blue")
        plt.hist(y_pred, bins=bins, alpha=0.5, label="Predictions", color="orange")
        plt.xlabel("Values")
        plt.ylabel("Frequency")
        plt.title(title)
        plt.legend(loc="upper right")
        plt.grid(True)
        plt.show()

    def plot_validation_histogram(self, X_val, y_val):
        if self.model is None:
            raise ValueError("The model has not been trained yet.")

        y_pred = self.predict(X_val)
        self.plot_predictions_histogram(y_val, y_pred, title="Validation: Ground Truth vs Predictions")