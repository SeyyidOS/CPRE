from torch.utils.tensorboard.writer import SummaryWriter
from sklearn.metrics import mean_squared_error

import xgboost as xgb


class XGBoostLearner:
    def __init__(self, params=None, log_dir="../runs/xgboost"):
        self.params = params if params else {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'eta': 0.1,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        }
        self.model = None
        self.log_dir = log_dir

    def train(self, X_train, y_train, X_val, y_val):
        train_data = xgb.DMatrix(X_train, label=y_train)
        val_data = xgb.DMatrix(X_val, label=y_val)

        self.model = xgb.train(
            self.params,
            train_data,
            num_boost_round=1000,
            evals=[(train_data, 'train'), (val_data, 'validation')],
            early_stopping_rounds=50,
            verbose_eval=100
        )

        writer = SummaryWriter(self.log_dir)

        for i, evals_log in enumerate(self.model.evals_result()['validation']['rmse']):
            writer.add_scalar('XGBoost/Validation_RMSE', evals_log, i)

        writer.close()

    def predict(self, X):
        if self.model is None:
            raise ValueError("The model has not been trained yet.")
        dmatrix = xgb.DMatrix(X)
        return self.model.predict(dmatrix)

    def evaluate(self, X_val, y_val):
        predictions = self.predict(X_val)
        rmse = mean_squared_error(y_val, predictions, squared=True)
        print(f"Validation RMSE: {rmse:.4f}")
        return rmse

    def load_model(self, model_path):
        self.model = xgb.Booster()
        self.model.load_model(model_path)

    def save_model(self, model_path):
        if self.model is None:
            raise ValueError("The model has not been trained yet.")
        self.model.save_model(model_path)