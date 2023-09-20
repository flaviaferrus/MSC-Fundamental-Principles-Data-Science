import pandas as pd
import numpy as np
from advanced import Logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple
from warnings import catch_warnings, simplefilter

logger = Logger()


#######################################################################
# CORE
#######################################################################

def encode_nan_as_strings(data: pd.DataFrame,
                          verbose: bool = False) -> pd.DataFrame:
    logger.debug("\t Encoding the nans as '{col_name}_nan' strings.")
    for col in data.columns:
        if data[col].dtype == 'category':
            new_categories = [f"{str(col).replace(' ', '_')}_nan"]
            with catch_warnings():
                simplefilter(action='ignore')
                try:
                    data[col].cat.add_categories(new_categories, inplace=True)
                except ValueError:
                    # it means the new_category is already in the dataset
                    # (it comes from the binary nan encoding, i.e. location)
                    pass
                data[col].fillna(new_categories[0], inplace=True)

    if verbose:
        logger.debug(f'\t\t There are {np.count_nonzero(pd.isna(data))} '
                     f'nans in the features')

    return data


def empty_strings_as_nan(features: pd.DataFrame,
                         verbose: bool = False) -> pd.DataFrame:
    """
    Encode the nans (and empty strings) of a categorical feature by
    assigning them a new category: "{feature_name}_nan".

    Parameters
    ----------
    features: pd.DataFrame
        Raw dataframe with nans (containing both features & labels)
    verbose: bool
        If True, prints info regarding the nan quantity
    Returns
    -------
    pd.DataFrame
        Dataframe with nans set as new categorical category for each feature

    """
    logger.debug("\t Encoding the empty strings as np.nan.")
    data = features.where(features != '', other=np.nan)

    if verbose:
        logger.debug(f'\t\t There are {np.count_nonzero(pd.isna(data))} '
                     f'nans in the features')
    return data


def data_normalization(data: pd.DataFrame) -> pd.DataFrame:
    try:
        data = StandardScaler().fit_transform(data.values)
    except ValueError:
        # empty dataframe
        pass
    return data


def binary_nan_encoding(data: pd.DataFrame) -> pd.DataFrame:
    logger.debug('\t Binary encoding into 2 categories: nan & not nan')
    for col in data.columns:
        new_categories = [f"{str(col).replace(' ', '_')}_notnan",
                          f"{str(col).replace(' ', '_')}_nan"]
        with catch_warnings():
            simplefilter(action='ignore')
            data[col].cat.add_categories(new_categories, inplace=True)
            data[col] = data[col].where(
                pd.isna(data[col]), other=new_categories[0])
            data[col] = data[col].where(
                ~pd.isna(data[col]), other=new_categories[-1])

    return data


def feature_extraction(data: pd.DataFrame) -> pd.DataFrame:
    """
    Several tasks regarding feature extraction and postprocessing performed:
    - Location is split in 'country', 'state' & 'city'
    - Replace the categories consisting in empty strings '' (for the
    categorical variables) by np.nan.

    Parameters
    ----------
    data

    Returns
    -------

    """
    logger.debug("\t Extracting features manually")
    # we separate location into the following 2 features
    cols: list = ['country', 'state', 'city']
    data[cols] = data['location'].str.split(',', expand=True).loc[:, :2]
    data.drop(columns=['location'], axis=1, inplace=True)

    return data


def split_and_scale(features: pd.DataFrame, labels: pd.DataFrame,
                    train_size: float = 0.7) \
        -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
                 pd.DataFrame, StandardScaler]:
    logger.debug(f"\t Splitting (using {np.round(train_size, 2)}, "
                 f"{np.round(1-train_size, 2)} split) and scaling the data.")
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, train_size=train_size, shuffle=True)
    scaler: StandardScaler = StandardScaler()

    # already scaled
    x_train: pd.DataFrame = pd.DataFrame(
        scaler.fit_transform(x_train),
        index=x_train.index,
        columns=x_train.columns)
    x_test: pd.DataFrame = pd.DataFrame(
        scaler.transform(x_test),
        index=x_test.index,
        columns=x_train.columns)

    return x_train, x_test, y_train, y_test, scaler
