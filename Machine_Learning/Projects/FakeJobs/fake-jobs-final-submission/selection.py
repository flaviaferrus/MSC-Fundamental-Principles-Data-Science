""""
Implement some routines for feature selection process.
"""

import lightgbm as lgbm
import pandas as pd
import numpy as np
from random import choices
from typing import Tuple
from advanced import Logger
from sklearn.feature_selection import RFE

logger = Logger()


######################################################################
# MAIN
######################################################################

def feature_selection(features: pd.DataFrame, labels: pd.DataFrame,
                      method: str = 'lgbm', manual_selection: bool = False) \
        -> Tuple[pd.DataFrame, list]:
    logger.debug(f"Performing automatic feature selection using {method}")
    if manual_selection:
        features = _add_noise_as_threshold(features)

    if method == 'lgbm':
        classifier, importance = lightgbm_selection(features, labels)
    else:
        raise ValueError(f"Unrecognized method to use for the feature"
                         f" selection: {method}.")

    if manual_selection:
        selected_features = __drop_meaningless_features(features, importance)
    else:
        selected_features = recursive_feature_elimination(
            features, labels, classifier)

    drop_feat_list: list = [c_ for c_ in features.columns if c_
                            not in selected_features.columns]

    return selected_features, drop_feat_list


######################################################################
# CORE
######################################################################


def lightgbm_selection(features: pd.DataFrame, labels: pd.DataFrame):

    # we order LGBM to fit/train and then find out which features
    # have been most important
    logger.debug("\t Training the LGBM classifier to find "
                 "out feature's importance")
    classifier = lgbm.LGBMClassifier()
    classifier.fit(features, labels)
    classifier.booster_.feature_importance()

    # importance of each attribute
    imp: pd.DataFrame = pd.DataFrame(
        {  # 'cols': features.columns,
         'feature_importance': classifier.feature_importances_})
    imp = imp.loc[imp.feature_importance > 0].sort_values(
        by=['feature_importance'], ascending=False)
    return classifier, imp


def recursive_feature_elimination(features, labels, classifier) -> pd.DataFrame:
    # create the RFE model and select 10 attributes
    rfe = RFE(classifier, n_features_to_select=len(features.columns) // 2)
    rfe = rfe.fit(features, labels)

    # summarize the ranking of the attributes
    ranking = pd.DataFrame({'cols': features.columns,
                            'fea_rank': rfe.ranking_})
    ranking = ranking.loc[ranking.fea_rank > 0].sort_values(
        by=['fea_rank'], ascending=True)

    selected_features = features[list(
        ranking[ranking.fea_rank == 1]['cols'].values)]

    return selected_features


######################################################################
# AUXILIARY
######################################################################

def __drop_meaningless_features(features, importance) -> pd.DataFrame:
    # TODO: implement
    return features


def _add_noise_as_threshold(dataframe: pd.DataFrame,
                            n_random_features: int = 3,
                            label: str = 'random') -> pd.DataFrame:
    """
    Add random noise as features in the given dataset.
    """

    # we define the different kind of noises to implement
    noises = choices(
        [np.random.uniform(-1., 1., size=(len(dataframe), 1)),
         np.random.normal(-1., 1., size=(len(dataframe), 1)),
         np.random.lognormal(-1., 1., size=(len(dataframe), 1)),
         np.random.weibull(np.random.randint(1, 9), size=(len(dataframe), 1)),
         ], k=n_random_features)

    # the choice of random noise is also random
    for i in range(n_random_features):
        dataframe = pd.concat(
            [dataframe,
             pd.DataFrame(noises[i], index=dataframe.index,
                          columns=[f"{label}_{i + 1}"])], axis=1)

    return dataframe
