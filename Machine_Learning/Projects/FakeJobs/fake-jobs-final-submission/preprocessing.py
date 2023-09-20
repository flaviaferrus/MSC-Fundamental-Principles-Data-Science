import pandas as pd
import numpy as np
from advanced import Logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Union
from warnings import catch_warnings, simplefilter
from static import binnan_features, numerical_features, synthetic_features
from auxiliary import reduce_mem_usage

logger = Logger()


#######################################################################
# MAIN
#######################################################################

def load_input(test: bool, verbose: bool = False) \
        -> Tuple[pd.DataFrame, Union[pd.DataFrame, None]]:
    folder: str = 'input'
    name: str = 'test.csv' if test else 'train.csv'

    if test:
        logger.info(f"Loading the input test dataset.")
    else:
        logger.debug(f"Loading the input train dataset.")
    data: pd.DataFrame = pd.read_csv(
        f'{folder}/{name}', sep=',', header=0, index_col=0)
    data = reduce_mem_usage(data)

    if test:
        data = data.rename(columns={
            'doughnuts_comsumption': 'required_doughnuts_comsumption'})

    features = data[[c_ for c_ in data.columns if c_ != 'fraudulent']]

    try:
        labels = data[['fraudulent']]
    except KeyError:
        # testing case
        labels = None

    if verbose:
        logger.debug(f'\t\t There are {np.count_nonzero(pd.isna(features))} '
                     f'nans in the features')

    # NECESSARY PREPROCESSING (nan removal & encoding into strings)
    original_features = location_extraction(features)
    original_features = empty_strings_as_nan(original_features)
    features = original_features.copy()
    # features[binnan_features] = binary_nan_encoding(features[binnan_features])
    features = encode_nan_as_strings(features)
    features[numerical_features] = data_normalization(
        features[numerical_features])
    features[synthetic_features] = add_nan_per_sample(original_features)

    return features, labels


#######################################################################
# CORE
#######################################################################

def encode_nan_as_strings(data: pd.DataFrame,
                          verbose: bool = False) -> pd.DataFrame:
    logger.debug("\t\t Encoding the nans as '{col_name}_nan' strings.")
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
    logger.debug("\t\t Encoding the empty strings as np.nan.")
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


def location_extraction(data: pd.DataFrame) -> pd.DataFrame:
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
    logger.debug(
        "\t\t Manually extracting 'location' -> 'country', 'state', 'city' ")
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


def add_nan_per_sample(features: pd.DataFrame) -> pd.DataFrame:
    nan_per_sample = np.count_nonzero(pd.isna(features), axis=1)
    return pd.DataFrame(nan_per_sample, columns=['nan_per_sample'])
