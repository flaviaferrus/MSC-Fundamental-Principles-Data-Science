import numpy as np
import pandas as pd
from typing import Tuple, Any
from xgboost import XGBClassifier
from auxiliary import load_input, save_data
from advanced import Logger
from preprocessing import empty_strings_as_nan, data_normalization, \
    binary_nan_encoding, encode_nan_as_strings  # feature_extraction
from encoding import BoWEncoding, OneHotEncoding
from training import Model, GridSearch
from reduction import DimensionalityReductor

logger = Logger()

bow_features = ['industry', 'function', 'company_profile', 'title', 'description']  # long categorical features
onehot_features = ['required_experience', 'required_education', 'employment_type']  # shorter categorical features
binnan_features = ['location', 'department', 'salary_range', 'requirements', 'benefits']  # binary encoded: '{col_name}_nan' & '{col_name}_notnan'
yesno_features = ['telecommuting', 'has_company_logo', 'has_questions']  # True or False features (of type 'category' however)
numerical_features = []  # ['required_doughnuts_comsumption']  # actually speaking the only one of type 'float' instead of 'category'

# TODO: perform 2 PCA: 1 for the BoW, 1 for the OneHot
# TODO: add the boolean features (boolean encoded), the binnan features (boolean encoded) & the numerical ones; outside any PCA calling them 'non_categorical' features

# TODO: once submitted: change between the former columns include or not include one of the above...
# TODO: even more hardcore, for the training discard some of the blocks BOW, ONE, BIN, NUM (most probable, between BOW and ONE there is the trouble)


def prepare_train_data() -> Tuple[Any, ...]:
    """
    Process data from the train.csv & save it
    """
    # We process input/train.csv
    features, labels = load_input(test=False)

    # features = feature_extraction(features)  # to process 'location'
    features = empty_strings_as_nan(features)
    features[binnan_features] = binary_nan_encoding(
        features[binnan_features])
    features = encode_nan_as_strings(features)
    features[numerical_features] = data_normalization(
        features[numerical_features])

    logger.debug("Advanced encoding")
    bow_encoder = BoWEncoding()
    onehot_encoder = OneHotEncoding()
    boolean_encoder = OneHotEncoding(name='Boolean')

    _bow_df = bow_encoder.fit_transform(
        features[bow_features])
    _onehot_df = onehot_encoder.fit_transform(
        features[onehot_features])
    _boolean_df = boolean_encoder.fit_transform(
        features[binnan_features + yesno_features])
    _doughnuts_df = features[numerical_features]

    logger.debug("Carrying out the dimensionality reduction")
    pca = DimensionalityReductor('PCA', prefix='PCA')

    features = pca.fit_transform(
        pd.concat([_bow_df, _onehot_df, _boolean_df, _doughnuts_df], axis=1),
        n_comps=220)

    # logger.debug("Merging all the features & renaming them properly")
    # _onehot_df.columns = [f"ONE{i_}" for i_ in range(_onehot_df.shape[1])]
    # _boolean_df.columns = [f"BIN{i_}" for i_ in range(_boolean_df.shape[1])]
    # _doughnuts_df.columns = [f"NUM{i_}" for i_ in range(_doughnuts_df.shape[1])]
    # features = pd.concat(
    #     [_bow_df, _onehot_df, _boolean_df, _doughnuts_df], axis=1)

    logger.debug("Saving the training data")
    save_data(features, 'x_train')
    save_data(labels, 'y_train')

    return bow_encoder, onehot_encoder, boolean_encoder, pca


def prepare_test_data(bow_encoder, onehot_encoder, boolean_encoder,
                      pca) -> None:
    features, _ = load_input(test=True)

    features = empty_strings_as_nan(features)
    features[binnan_features] = binary_nan_encoding(
        features[binnan_features])
    features = encode_nan_as_strings(features)
    features[numerical_features] = data_normalization(
        features[numerical_features])

    logger.info("Advanced encoding")
    _bow_df = bow_encoder.transform(
        features[bow_features])
    _onehot_df = onehot_encoder.transform(
        features[onehot_features])
    _boolean_df = boolean_encoder.transform(
        features[binnan_features + yesno_features])
    _doughnuts_df = features[numerical_features]

    logger.info("Carrying out the dimensionality reduction")

    features = pca.transform(
        pd.concat([_bow_df, _onehot_df, _boolean_df, _doughnuts_df], axis=1))

    # logger.info("Merging all the features & renaming them properly")
    # _onehot_df.columns = [f"ONE{i_}" for i_ in range(_onehot_df.shape[1])]
    # _boolean_df.columns = [f"BIN{i_}" for i_ in range(_boolean_df.shape[1])]
    # _doughnuts_df.columns = [f"NUM{i_}" for i_ in range(_doughnuts_df.shape[1])]
    # features = pd.concat(
    #     [_bow_df, _onehot_df, _boolean_df, _doughnuts_df], axis=1)

    logger.info("Saving the testing data")
    save_data(features, 'x_test')


def find_best_model(time_consuming: bool = False) -> dict:
    """
    Find the best hyperparameters using a CV GridSearch.
    """
    features: pd.DataFrame = pd.read_csv('output/x_train.csv', index_col=0)
    labels: pd.DataFrame = pd.read_csv('output/y_train.csv', index_col=0)
    best_params = GridSearch(time_consuming=time_consuming).fit(
        features, labels)
    return best_params


def train_model(**kwargs) -> None:
    """
    Once the best model's hyperparameters are found, we train that model
    (with the data already processed) & apply it for inference
    (& save both the model and the results).

    :param **kwargs
    """
    x_train: pd.DataFrame = pd.read_csv('output/x_train.csv', index_col=0)
    y_train: pd.DataFrame = pd.read_csv('output/y_train.csv', index_col=0)
    x_test: pd.DataFrame = pd.read_csv('output/x_test.csv', index_col=0)

    classifier = XGBClassifier(**kwargs)
    Model(classifier).fit(x_train, y_train)

    y_hat: np.ndarray = Model().predict(x_test)

    logger.info("Saving the obtained predictions")
    y_hat_df: pd.DataFrame = pd.DataFrame(
        y_hat, columns=['Category'], index=x_test.index)
    y_hat_df.index.name = 'Id'
    y_hat_df.to_csv('output/y_hat.csv')


def hard_coded_training() -> None:
    logger.debug("Loading the train/test processed data")
    x_train: pd.DataFrame = pd.read_csv('output/x_train.csv', index_col=0)
    y_train: pd.DataFrame = pd.read_csv('output/y_train.csv', index_col=0)
    x_test: pd.DataFrame = pd.read_csv('output/x_test.csv', index_col=0)

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

    logger.debug("Predicting the outcome")

    # Predict training set:
    y_hat = alg.predict(x_test)

    logger.info("Saving the obtained predictions")
    y_hat_df: pd.DataFrame = pd.DataFrame(
        y_hat, columns=['Category'], index=x_test.index)
    y_hat_df.index.name = 'Id'
    y_hat_df.to_csv('output/y_hat.csv')


if __name__ == '__main__':  # to indicate the program what to execute when ran
    # # it must be firstly ran
    processors = prepare_train_data()
    prepare_test_data(*processors)

    # _best_params = find_best_model(time_consuming=False)  # look best params
    # train_model(n_jobs=-1, **_best_params)  # n_jobs = nº of cores to use

    # # Or manually typing the best params of XGB found so far...
    # _best_params = {'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 1300}
    # train_model(n_jobs=-1, **_best_params)
    hard_coded_training()

    # # or we could read them from the serialized file
    # import pickle
    # _best_params: dict = pickle.load(
    #     open(f"output/best_params.pkl", 'rb))
    # train_model(n_jobs=-1, **_best_params)
