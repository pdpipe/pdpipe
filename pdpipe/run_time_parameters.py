"""Defines run-time parameterization capabilities."""

from typing import Any, Callable, Optional
import pandas as pd

from .shared import POS_ARG_MISMTCH_PAT


class DynamicParameter:
    """
    A dynamicly determined parameter, determined at runtime.

    This class represents a dynamic parameter that is decided at fit time when
    there is access to the given dataframe. This allows the parameter to be
    decided at "run time" depending on the given dataframe

    Parameters
    ----------
    parameter_selector : Callable
        Performs logic in order to decide on a given parameter at run time,
        using the given DataFrame and optionally y label series.
    fittable : bool
        Denotes whether the parameter can be fitted, or should perform the
        deciding logic every time.
    """
    def __init__(
        self,
        parameter_selector: Callable,
        fittable: Optional[bool] = True,  # if False, fit in each use
    ) -> None:
        self._parameter_selector = parameter_selector
        self._fittable = fittable
        self._parameter: Any

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
    ):
        """
        Compute and return the value of this dynamic parameter.

        Parameters
        ----------
        X : pandas.DataFrame
            Input dataframe upon which the parameter selector decides which
            parameter to use.
        y : pandas.Series
            Label column, if separated from X dataframe at an earlier stage.
            Optional.

        Returns
        -------
        object
            The dynamically determined value of this parameter.
        """
        try:
            self._parameter = self._parameter_selector(X, y)
        except TypeError as e:
            if len(POS_ARG_MISMTCH_PAT.findall(str(e))) > 0:
                self._parameter = self._parameter_selector(X)
            else:
                raise TypeError('Parameter selector must be callable') from e
        return self._parameter

    def transform(self) -> object:
        """
        Return the fitted value of this dynamic parameter.

        Returns
        -------
        object
            The fitted value of this dynamic parameter.
        """
        if not hasattr(self, '_parameter'):
            raise AttributeError('Parameter not fitted')  # pragma: no cover
        return self._parameter

    def __call__(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        *args,
        **kwargs,
    ):
        if self._fittable and hasattr(self, '_parameter'):
            return self.transform()
        else:
            return self.fit_transform(X, y)


def dynamic(
    parameter_selector: Callable,
    fittable: Optional[bool] = True,
) -> DynamicParameter:
    """
    Return a dynamic parameter.

    Parameters
    ----------
    parameter_selector : callable
        The function or callable implementing the parameter value selection
        logic based on an input dataframe.
    fittable : bool
        Determines whether this dynamic parameter's value should be determined
        once on fit - in which case all future transforms get the saved value
        - or, alternatively, the value-selection logic should be performed on
        both every fit-transform and every transform operation.

    Returns
    -------
    DynamicParameter
        An object with the parameter selecting logic.
    """
    return DynamicParameter(parameter_selector, fittable)
