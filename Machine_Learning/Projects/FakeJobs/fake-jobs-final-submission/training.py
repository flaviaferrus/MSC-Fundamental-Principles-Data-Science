import pickle

import pandas as pd
from xgboost import XGBClassifier
from advanced import Logger, SingletonMeta
from preprocessing import split_and_scale
from typing import Any, Dict
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from multiprocessing import cpu_count

n_cores: int = cpu_count()
logger = Logger()


class Model(metaclass=SingletonMeta):
    def __init__(self, classifier: Any = None) -> None:
        """
        Wrapper to fit & predict the passed classifier. The model
        is trained with all the available data

        Parameters
        ----------
        classifier: Any
            Classifier to be trained. Ideally it should come from
            a GridSearch CV (or somehow it should be the one
            w/ the best hyperparameters)
        """
        assert classifier is not None, \
            "There is no initialized Model instance yet: then, " \
            "the classifier parameter must be passed for the first time!"

        # instead, perform a RandomSearch w/ CV to fine-tune
        self.classifier: Any = classifier
        self.scaler: StandardScaler = StandardScaler()

    # TODO: implement 10-fold CV for model's fitting with early stopping

    def fit(self, features: pd.DataFrame, labels: pd.DataFrame) -> None:

        x_train, x_val, y_train, y_val, self.scaler = split_and_scale(
            features, labels)
        x_train, y_train = x_train.values, y_train.values.reshape((-1, ))
        x_val, y_val = x_val.values, y_val.values.reshape((-1, ))

        logger.debug(f"Training the model")
        self.classifier.fit(
            x_train, y_train,
            eval_set=[(x_train, y_train), (x_val, y_val)],
            early_stopping_rounds=20, verbose=1)

    def predict(self, data: pd.DataFrame):
        logger.info(f"Using the trained model for actual inference.")
        return self.classifier.predict(self.scaler.transform(data))


class GridSearch(metaclass=SingletonMeta):
    def __init__(self, method: str = 'XGB', time_consuming: bool = False):
        self.method: str = method
        self._cv_scaler: StandardScaler = StandardScaler()
        self._time_consuming: bool = time_consuming
        if self.method == 'XGB':
            # estimator whose best hyperparameters have to be found
            self.estimator = XGBClassifier(
                objective='binary:logistic', nthread=-1, seed=42)
            # hyperparameters to explore and try
            if not self._time_consuming:
                self.parameters = {
                    'max_depth': [6],  # range(6, 9, 2),
                    'n_estimators': range(750, 905, 50),
                    'learning_rate': [0.1]
                }
            else:
                self.parameters = {
                    'max_depth': range(4, 9, 1),
                    'n_estimators': range(700, 2005, 100),
                    'learning_rate': [0.05, 0.1]  #
                }

        else:
            raise ValueError("Unrecognized choice of estimator "
                             f"for the GridSearch: {method}")

        # the best hyperparameters will be chosen using the ROC AUC metric
        # to compare the results of 10-fold cross-validation
        self.grid_search = GridSearchCV(
            estimator=self.estimator,
            param_grid=self.parameters,
            scoring='roc_auc',
            n_jobs=2 * n_cores // 3,
            cv=5,
            verbose=2
        )

    def fit(self, features: pd.DataFrame, labels: pd.DataFrame) -> Dict:
        # logger message
        logger_message: str = "Performing a grid search to find the best " \
                              "hyperparameters of {self.method}"
        logger_message += " exhaustively. WAIT A LOT!!!" if \
            self._time_consuming else ". WAIT!"
        logger.debug(logger_message)

        # we begin to find the best hyperparameters
        self.grid_search.fit(self._cv_scaler.fit_transform(features),
                             labels.values.reshape((-1, )))
        print(f"\t The best hyperparameters found were: \n",
              self.grid_search.best_params_, "\n Saving them... \n",
              flush=True)

        pickle.dump(self.grid_search.best_params_,
                    open(f"output/best_params.pkl", "wb"))

        logger.debug(f"\t The mean cross-validated score of "
                     f"the best estimator (in the test sets) is: "
                     f"{self.grid_search.best_score_}")
        return self.grid_search.best_params_
