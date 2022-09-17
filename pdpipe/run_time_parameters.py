from typing import Any, Callable, Optional
import pandas as pd

from .shared import POS_ARG_MISMTCH_PAT


class DynamicParameter:
    """
    This class represents a dynamic parameter that is decided at fit time when
    there is access to the given dataframe. This allows the parameter to be
    decided at "run time" depending on the given dataframe

    Parameters
    ----------
    parameter_selector: Callable, preforms logic in order to decide on a given
     parameter at run time, using the given DataFrame and optionally y label
     series.
    fittable: boolean denoting whether the parameter can be fitted, or should
     perform the deciding logic every time.
    """
    def __init__(
            self,
            parameter_selector: Callable,
            fittable: Optional[bool] = True,  # if False, fit in each use
    ) -> None:
        self._parameter_selector = parameter_selector
        self._fittable = fittable
        self._parameter: Any

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
        except TypeError as e:
            if len(POS_ARG_MISMTCH_PAT.findall(str(e))) > 0:
                self._parameter = self._parameter_selector(X)
            else:
                raise TypeError('Parameter selector must be callable') from e
        return self._parameter

    def transform(self):
        """
        Transform returns the fitted parameter based on given dataframe
        """
        if not hasattr(self, '_parameter'):
            raise AttributeError('Parameter not fitted')  # pragma: no cover
        return self._parameter

    def __call__(self,
                 X: pd.DataFrame,
                 y: Optional[pd.Series] = None,
                 *args, **kwargs):
        if self._fittable and hasattr(self, '_parameter'):
            return self.transform()
        else:
            return self.fit_transform(X, y)


def dynamic(parameter_selector: Callable, fittable: Optional[bool] = True) \
        -> DynamicParameter:
    """
    Factory for the DynamicParameter class. This is used
    Parameters
    ----------
    parameter_selector: callable holding the logic on how to select a certain
     parameter based on given df
    fittable: boolean denoting whether the parameter can be fitted, or should
     perform the deciding logic every time.
    Returns DynamicParameter object with the parameter selecting logic
    -------
    """
    return DynamicParameter(parameter_selector, fittable)
