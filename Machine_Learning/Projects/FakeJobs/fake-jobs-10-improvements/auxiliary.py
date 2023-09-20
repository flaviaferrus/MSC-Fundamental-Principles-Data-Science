import os
import pandas as pd
import numpy as np
from typing import Tuple, Union
from advanced import Logger

logger = Logger()


def reduce_mem_usage(df: pd.DataFrame, silent: bool = True) -> pd.DataFrame:
    """
    Iterate through all the columns of a dataframe and modify the data type
    to reduce memory usage
    """
    # noinspection PyArgumentList
    start_mem = df.memory_usage().sum() / 1024 ** 2

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object and col_type != 'category':
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and \
                        c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and \
                        c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and \
                        c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and \
                        c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                # We avoid using float16...
                if c_min > np.finfo(np.float32).min and \
                        c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    # noinspection PyArgumentList
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if not silent:
        logger.debug(
            f"Reducing memory usage of the dataset: decreased a"
            f"{np.round(100 * (start_mem - end_mem) / start_mem, 2)} % ")

    return df


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
        labels = None

    if verbose:
        logger.debug(f'\t\t There are {np.count_nonzero(pd.isna(features))} '
                     f'nans in the features')

    return features, labels


# for the data-exploration.ipynb notebook...
def compute_nan_statistics(data: pd.DataFrame) -> pd.DataFrame:
    statistics = pd.DataFrame()

    # we add the percentage of samples affected by nan for each variable...
    statistics['% samples w/ nan'] = 100 * data.apply(
        lambda x: pd.isna(x)).sum(axis=0) / len(data)

    # now we generate columns indicating how likely is each variable
    # to present nan in a sample where are n nans in total,
    # for n=2,...,N_max where N_max is the number of features that
    # presents nan at least for some sample.

    n_max = np.count_nonzero(statistics > 0.)
    for n in range(2, n_max):
        more_than_one_nan = np.where(pd.isna(data), 1, 0)
        mask = (np.sum(more_than_one_nan, axis=1) - (n-1)).reshape(-1, 1)
        more_than_one_nan = np.where(more_than_one_nan * mask > 0.,
                                     more_than_one_nan, 0.)
        more_than_one_nan = 100 * pd.DataFrame(data=more_than_one_nan,
                                               columns=data.columns,
                                               index=data.index).mean(axis=0)
        statistics[f"% samples w/ {n} nan"] = more_than_one_nan

    return statistics


def save_data(data: pd.DataFrame, file_name: str) -> None:
    if file_name == 'x_test':
        logger.info(f"\t Saving the cleaned {file_name}. WAIT")
    else:
        logger.debug(f"\t Saving the cleaned {file_name}. WAIT")

    if not os.path.isdir('output/'):
        os.mkdir('output/')
    data.to_csv(f"output/{file_name}.csv", index=True)
