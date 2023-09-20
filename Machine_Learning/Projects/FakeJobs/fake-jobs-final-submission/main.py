import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Tuple, Any, List
from xgboost import XGBClassifier, plot_importance
from auxiliary import save_data
from advanced import Logger
from preprocessing import load_input
from encoding import BoWEncoding, load_hotwords  # OneHotEncoding
from static import synthetic_features
from training import Model, GridSearch
from reduction import DimensionalityReductor

logger = Logger()

# IDEA: preprocess the data in the following fashion:
# - To use Bag of Words with all the long categorical features.
# - To apply OneHot encoding to the rest of them (even those with lot of nans)
# - Add the nan_per_sample_feature
# - To perform a Word Embedding with the Bag of Words info.
#  Then simply merge the features coming from the OneHot (or alternatively use a PCA to merge and summarize)

# Check the following guide to know how the text is properly preprocessed:
# https://practicaldatascience.co.uk/machine-learning/how-to-detect-fake-news-with-machine-learning


def prepare_train_data(bow_columns: List[str],
                       load_hotwords_file: bool = False) -> Tuple[Any, ...]:
    """
    Process data from the train.csv & save it
    """
    # We process input/train.csv
    features, labels = load_input(test=False)

    logger.debug(f"Extracting the hot words of the features: "
                 f"{', '.join(bow_columns)}")
    _bow_series: pd.Series = load_hotwords(
        features[bow_columns], test=False, load=load_hotwords_file)

    logger.debug("Advanced encoding")
    bow_encoder = BoWEncoding()
    # onehot_encoder = OneHotEncoding()
    # boolean_encoder = OneHotEncoding(name='Boolean')

    _bow_df = bow_encoder.fit_transform(_bow_series)
    # _onehot_df = onehot_encoder.fit_transform(
    #     features[onehot_features])
    # _boolean_df = boolean_encoder.fit_transform(
    #     features[binnan_features + yesno_features])
    # _doughnuts_df = features[numerical_features]

    logger.debug("Carrying out the dimensionality reduction")
    pca = DimensionalityReductor('PCA', prefix='PCA')

    # features = pca.fit_transform(
    #     pd.concat([_bow_df, _onehot_df, _boolean_df, _doughnuts_df], axis=1),
    #     n_comps=220)

    # According to my feature importance study, just the BoW features are
    # interesting (and more correlated than random noise), except also
    # the synthetic feature 'nan_per_sample'...
    _bow_df = pca.fit_transform(_bow_df, n_comps=220)
    features = pd.concat([_bow_df, features[synthetic_features]], axis=1)

    logger.debug("Saving the training data")
    save_data(features, 'x_train')
    save_data(labels, 'y_train')

    return bow_encoder, pca


def prepare_test_data(bow_encoder, pca,
                      bow_columns: List[str], load_hotwords_file: bool = False) -> None:
    features, _ = load_input(test=True)

    logger.debug(f"Extracting the hot words of the features: "
                 f"{', '.join(bow_columns)}")
    _bow_series: pd.Series = load_hotwords(
        features[bow_columns], test=True, load=load_hotwords_file)

    logger.debug("Advanced encoding")
    _bow_df = bow_encoder.transform(_bow_series)

    logger.debug("Carrying out the dimensionality reduction")
    _bow_df = pca.transform(_bow_df, n_comps=220)
    features = pd.concat([_bow_df, features[synthetic_features]], axis=1)

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
    from sklearn.metrics import f1_score
    classifier = XGBClassifier(**kwargs, objective='binary:hinge',
                               metrics=['auc', f1_score])
    Model(classifier).fit(x_train, y_train)

    y_hat: np.ndarray = Model().predict(x_test)

    logger.info("Saving the obtained predictions")
    y_hat_df: pd.DataFrame = pd.DataFrame(
        y_hat, columns=['Category'], index=x_test.index)
    y_hat_df.index.name = 'Id'
    y_hat_df.to_csv('output/y_hat.csv')


def hard_coded_training(plot: bool = False) -> None:
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

    if plot:
        plot_importance(alg)
        plt.savefig('output/x_importance_training.png')

    logger.debug("Predicting the outcome")

    # Predict training set:
    y_hat = alg.predict(x_test)

    logger.info("Saving the obtained predictions")
    y_hat_df: pd.DataFrame = pd.DataFrame(
        y_hat, columns=['Category'], index=x_test.index)
    y_hat_df.index.name = 'Id'
    y_hat_df.to_csv('output/y_hat.csv')


def main() -> None:
    # # it must be firstly ran
    bow_columns = ['industry', 'function', 'company_profile', 'title',
                   'description']  # long categorical features
    # processors = prepare_train_data(bow_columns)
    # prepare_test_data(*processors, bow_columns=bow_columns)

    # _best_params = find_best_model(time_consuming=False)  # look best params
    # train_model(n_jobs=-1, **_best_params)  # n_jobs = nº of cores to use

    # # Or manually typing the best params of XGB found so far...
    _best_params = {'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 1300}
    train_model(n_jobs=-1, **_best_params)
    # hard_coded_training(plot=True)

    # # or we could read them from the serialized file
    # import pickle
    # _best_params: dict = pickle.load(
    #     open(f"output/best_params.pkl", 'rb))
    # train_model(n_jobs=-1, **_best_params)


if __name__ == '__main__':  # to indicate the program what to execute when ran
    main()
