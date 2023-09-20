"""
Implement some dimensionality reduction techniques.
"""
import numpy as np
import pandas as pd
from typing import Any
from sklearn.decomposition import KernelPCA, PCA
from advanced import Logger
import multiprocessing

n_cores: int = multiprocessing.cpu_count()
logger = Logger()


######################################################################
# CORE CLASSES
######################################################################


class DimensionalityReductor:
    def __init__(self, method: str = 'KPCA', prefix: str = 'COL'):
        self.method: str = method
        self._prefix: str = prefix
        self.reductor: Any = None

    def fit_transform(self, features: pd.DataFrame,
                      n_comps: int = None) -> pd.DataFrame:
        self._define_reductor_to_use(features, n_comps)
        logger.debug(
            f"\t Training a {self.method}. WAIT")
        # noinspection PyUnresolvedReferences
        reduced_: np.ndarray = self.reductor.fit_transform(features)

        return pd.DataFrame(
            reduced_, index=features.index,
            columns=[f'{self._prefix}{i}' for
                     i in range(self.reductor.n_components)])

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"\t Applying the trained {self.method}")
        return pd.DataFrame(
            self.reductor.transform(features), index=features.index,
            columns=[f'{self._prefix}{i}' for
                     i in range(self.reductor.n_components)])

    def _define_reductor_to_use(self, features, n_comps) -> None:
        if self.method == 'KPCA':  # default
            self.reductor = KernelPCA(
                n_components=n_comps if n_comps is not None else 300)
            # n_jobs=n_cores // 2)

        elif self.method == 'PCA':
            if n_comps is None and len(features) >= len(features.columns):
                self.reductor = PCA(n_components='mle')
            elif n_comps is None and len(features) < len(features.columns):
                self.reductor = PCA(n_components=len(features.columns))
            else:
                self.reductor = PCA(min(n_comps, len(features.columns)))

        else:
            raise ValueError(f"Unknown dimensionality reductor"
                             f" passed: {self.method}")
