from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import numpy as np

class RNN_Transform_Wrap(BaseEstimator, TransformerMixin):
    """ Wrapper to apply scaling transformation to 3d shaped array required
        for LSTM models """
    @staticmethod
    def _conv_3d_to_2d(X):
        if X is None:
            return None
        if X.ndim == 2: # Need to expand single variable
            X = np.expand_dims(X, 2)
        return np.vstack([np.squeeze(x, 0) for x in np.vsplit(X, X.shape[0])])

    @staticmethod
    def _conv_2d_to_3d(X, n_timesteps):
        if X is None:
            return None
        return np.vstack([np.expand_dims(x, 0) for x in np.vsplit(X, X.shape[0]/n_timesteps)])

    def __init__(self, transformer, *args, **kwargs):
        self.transformer = transformer(*args, **kwargs)

    def get_params(self, deep=True):
        out = self.transformer.get_params(deep=deep)
        out['transformer'] = self.transformer.__class__
        return out

    def set_params(self, **params):
        self.transformer = params.pop('transformer', None)()
        self.transformer.set_params(**params)
        return self

    def transform(self, X, *args, **kwargs):
        n_timesteps = X.shape[1]
        out = self.transformer.transform(self._conv_3d_to_2d(X), *args, **kwargs)
        return self._conv_2d_to_3d(out, n_timesteps)

    def inverse_transform(self, X, *args, **kwargs):
        n_timesteps = X.shape[1]
        out = self.transformer.inverse_transform(self._conv_3d_to_2d(X), *args, **kwargs)
        return self._conv_2d_to_3d(out, n_timesteps)

    def fit(self, X, y=None, *args, **kwargs):
        self.transformer.fit(self._conv_3d_to_2d(X), self._conv_3d_to_2d(y), *args, **kwargs)
        return self
