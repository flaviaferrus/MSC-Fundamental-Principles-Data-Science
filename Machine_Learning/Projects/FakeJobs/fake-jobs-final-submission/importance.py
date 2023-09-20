import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier, plot_importance
from xgboost import plot_importance
from typing import Tuple
from warnings import catch_warnings, simplefilter
from auxiliary import save_data
from advanced import Logger
from static import onehot_features, binnan_features, yesno_features, \
    numerical_features
from preprocessing import empty_strings_as_nan, data_normalization, \
    binary_nan_encoding, encode_nan_as_strings, \
    add_nan_per_sample, load_input  # feature_extraction
from encoding import OneHotEncoding


logger = Logger()


def importance_from_lgbm(x_train, y_train) -> None:
    lgbm_model = LGBMClassifier()
    lgbm_model.fit(x_train, y_train.values.ravel())
    with catch_warnings():
        simplefilter('ignore')
        plot_importance(lgbm_model, figsize=(10, 9))
    plt.show()


def importance_from_xgboost(x_train, y_train) -> None:
    from xgboost.sklearn import XGBClassifier

    logger.debug("Defining the classifier & learning task")

    alg = XGBClassifier(
        learning_rate=0.1,
        min_child_weight=1,
        max_depth=6,
        n_estimators=1300,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        nthread=-1,
        reg_alpha=0.01,
        scale_pos_weight=1,
        eval_metric='auc',
        seed=27)

    import xgboost as xgb
    xgtrain = xgb.DMatrix(x_train.values, label=y_train.values)
    xgb_param = alg.get_xgb_params()
    cvresult = xgb.cv(xgb_param, xgtrain,
                      num_boost_round=alg.get_params()['n_estimators'],
                      nfold=5,
                      metrics='auc',
                      early_stopping_rounds=50)
    logger.warning(f"Best N estimators found: {cvresult.shape[0]}")
    print(cvresult)
    alg.set_params(n_estimators=cvresult.shape[0])

    logger.debug("Training the classifier")
    # Fit the algorithm on the data
    alg.fit(x_train, y_train, verbose=2)
    logger.debug("Plotting the features importance")
    with catch_warnings():
        simplefilter('ignore')
        plot_importance(alg)
    plt.show()


def _load_non_bow_features(original_features: pd.DataFrame, load: bool = False) -> pd.DataFrame:
    if load:
        logger.debug("Loading already processed features dataset")
        return pd.read_csv('output/x_importance.csv', index_col=0)
    # features = feature_extraction(features)  # to process 'location'
    original_features = empty_strings_as_nan(original_features)
    features = original_features.copy()
    features[binnan_features] = binary_nan_encoding(
        features[binnan_features])
    features = encode_nan_as_strings(features)
    features[numerical_features] = data_normalization(
        features[numerical_features])

    logger.debug("Advanced encoding")
    onehot_encoder = OneHotEncoding()
    boolean_encoder = OneHotEncoding(name='Boolean')

    _onehot_df = onehot_encoder.fit_transform(
        features[onehot_features])
    _onehot_df.columns = [f"ONE_{c_}" for c_ in _onehot_df.columns]
    _boolean_df = boolean_encoder.fit_transform(
        features[binnan_features + yesno_features])
    _doughnuts_df = features[numerical_features]

    new_features = pd.concat([_onehot_df, _boolean_df, _doughnuts_df], axis=1)
    new_features = _add_random_noise(new_features)
    new_features = pd.concat(
        [new_features, add_nan_per_sample(original_features)], axis=1)
    save_data(new_features, 'x_importance')
    return new_features


def _load_processed_datasets() -> Tuple[pd.DataFrame, pd.DataFrame]:
    x_train: pd.DataFrame = pd.read_csv('output/x_train.csv', index_col=0)
    x_train = x_train[
        [c_ for c_ in x_train.columns if not str(c_).startswith('BOW')]]
    y_train: pd.DataFrame = pd.read_csv('output/y_train.csv', index_col=0)
    return x_train, y_train


def _add_random_noise(data: pd.DataFrame, n_cols: int = 3,
                      rand_integers: bool = True) -> pd.DataFrame:
    if rand_integers:
        from numpy.random import randint
        rand_arr_: np.ndarray = randint(0, 2, size=(len(data), n_cols))
    else:
        from numpy.random import randn
        rand_arr_: np.ndarray = randn(len(data), n_cols)
    random_df: pd.DataFrame = pd.DataFrame(
        rand_arr_,
        columns=[f'RANDOM{i+1}' for i in range(n_cols)])
    data = pd.concat([data, random_df], axis=1)
    return data


def main():
    features, labels = load_input(test=False)
    features = _load_non_bow_features(features, load=False)
    # importance_from_lgbm(features, labels)
    importance_from_xgboost(features, labels)


if __name__ == '__main__':
    main()
