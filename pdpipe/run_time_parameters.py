from typing import Callable, Optional
import pandas as pd


def dynamic(parameter_selector: Callable):
    return DynamicParameter(parameter_selector)


class DynamicParameter:
    """
    This class represents a dynamic parameter that is decided at fit time when
    there is access to the given dataframe. This allows the parameter to be
    decided at "run time" depending on the given dataframe
    """
    def __init__(self, parameter_selector: Callable):
        self._parameter_selector = parameter_selector
        self.is_fitted = False
        self._parameter = None

    def fit_transform(self,
                      X: pd.DataFrame,
                      y: Optional[pd.Series] = None):
        """

        Parameters
        ----------
        X: Input dataframe upon which the parameter selector decides which
           parameter to use
        y: optional label column, if separated from X dataframe at an earlier
           stage

        Returns
        -------

        """
        try:
            self._parameter = self._parameter_selector(X, y)
        except TypeError:
            raise TypeError('Parameter selector must be callable')

        self.is_fitted = True
        return self._parameter

    def transform(self):
        """
        Returns
        -------
        Transform returns the fitted parameter based on given dataframe
        """
        if self._parameter is None:
            raise AttributeError('Parameter not fitted')
        return self._parameter

    def __call__(self,
                 X: pd.DataFrame,
                 y: Optional[pd.Series] = None,
                 *args, **kwargs):
        if self.is_fitted:
            return self.transform()
        else:
            return self.fit_transform(X, y)
