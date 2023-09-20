import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder
from warnings import catch_warnings, simplefilter
from advanced import Logger

logger = Logger()

# For + information on BoW, check: https://towardsdatascience.com/
# a-guide-to-encoding-text-in-python-ef783e50f09e


class BoWEncoding:
    def __init__(self):
        self.bow_encoder = CountVectorizer(stop_words='english')
        # ngram_range=(2, 2) not set bc we got ~500.000 features!

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        logger.debug("\t Training & applying Bag of Words encoding")
        string_series_ = _set_dataframe_for_bow(data)
        bow_arr_ = self.bow_encoder.fit_transform(string_series_).toarray()
        bow_df_ = pd.DataFrame(
            bow_arr_, columns=self.bow_encoder.get_feature_names_out())
        return bow_df_.astype(np.int8)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        logger.info("\t Applying Bag of Words encoding")
        string_series_ = _set_dataframe_for_bow(data)
        bow_arr_ = self.bow_encoder.transform(string_series_).toarray()
        bow_df_ = pd.DataFrame(
            bow_arr_, columns=self.bow_encoder.get_feature_names_out())
        return bow_df_.astype(np.int8)


def _set_dataframe_for_bow(data: pd.DataFrame) -> pd.Series:
    # we transform all the features into a one feature: large string
    document_series: pd.Series = data.apply(
        lambda x: " ".join(x.astype(str)), axis=1)
    return document_series


class OneHotEncoding:
    def __init__(self, name: str = 'OneHot'):
        self._encoder: SklearnOneHotEncoder = SklearnOneHotEncoder()
        self.__name: str = name

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        logger.debug(f"\t Training & applying {self.__name} encoding")
        onehot_arr_ = self._encoder.fit_transform(data).toarray()
        onehot_df = pd.DataFrame(
            onehot_arr_, columns=self._encoder.get_feature_names_out())
        return onehot_df.astype(np.int8)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"\t Applying {self.__name} encoding")
        onehot_arr_ = self._encoder.transform(data).toarray()
        onehot_df = pd.DataFrame(
            onehot_arr_, columns=self._encoder.get_feature_names_out())
        return onehot_df.astype(np.int8)


def _onehot_arr_to_dframe(arr_: np.ndarray, columns: list) -> pd.DataFrame:
    with catch_warnings():
        simplefilter(action='ignore', category=FutureWarning)
        return pd.DataFrame(arr_, columns=columns).astype(np.int8)
