"""PdPipeline stages that transform the optional label column."""

from typing import Optional, Iterable

import pandas
from pdpipe.core import PdPipelineStage

from .exceptions import (
    PipelineInitializationError,
    UnexpectedPipelineMethodCallError,
)
from .util import _LBL_PHOLDER_PREDICT


class _SkipOnLabelPlaceholderPredict:

    def __init__(self, skip_cond: Optional[callable] = None) -> None:
        self.skip_cond = skip_cond

    def __call__(
        self,
        X: pandas.DataFrame,
        y: Optional[pandas.Series] = None,
    ) -> bool:  # pylint: disable=R0201,W0613
        try:
            if y.iloc[0] == _LBL_PHOLDER_PREDICT:
                return True
        except (AttributeError, IndexError):
            pass
        if self.skip_cond is not None:
            return self.skip_cond(X, y)
        return False


class DropLabelsByValues(PdPipelineStage):
    """A pipeline stage that drop values from the input label series.

    Parameters
    ----------
    in_set : iterable of object, optional
        Drop all labels in the input set of values.
    in_ranges : iterable of iterables of scalars, optional
        Drop all labels in the input ranges of values.
    not_in_set : iterable of object, optional
        Drop all labels not in the input set of values.
    not_in_ranges : iterable of iterables of scalars, optional
        Drop all labels not in the input ranges of values.


    Examples
    --------
    >>> import pandas as pd; import pdpipe as pdp;
    >>> data = [[3.2, 31], [7.2, 33], [12.1, 28]]
    >>> X = pd.DataFrame(data, [1,2,3], ["ph","temp"])
    >>> y = pd.Series(["acd", "alk", "alk"])
    >>> drop_labels = pdp.DropLabelsByValues(in_set=["acd"])
    >>> X, y = drop_labels(X, y)
    >>> y
    2    alk
    3    alk
    dtype: object
    >>> X
         ph  temp
    2   7.2    33
    3  12.1    28
    """

    def __init__(
        self,
        in_set: Optional[Iterable[object]] = None,
        in_ranges: Optional[Iterable[Iterable[object]]] = None,
        not_in_set: Optional[Iterable[object]] = None,
        not_in_ranges: Optional[Iterable[Iterable[object]]] = None,
        **kwargs: object,
    ) -> None:
        self.in_set = in_set
        self.in_ranges = in_ranges
        self.not_in_set = not_in_set
        self.not_in_ranges = not_in_ranges
        skipi = _SkipOnLabelPlaceholderPredict()
        if 'skip' in kwargs:
            skipi.skip_cond = kwargs.pop('skip')
        super_kwargs = {
            'desc': "Drop rows by label values",
            'skip': skipi,
        }
        super_kwargs.update(**kwargs)
        super().__init__(**super_kwargs)

    def _prec(self, X, y):
        return y is not None

    def _transform(self, X, verbose):
        raise UnexpectedPipelineMethodCallError(  # pragma: no cover
            "DropLabelsByValues._transform() is not expected to be called!")

    def _transform_Xy(self, X, y, verbose):
        post_y = y
        if self.in_set is not None:
            post_y = post_y.loc[~ post_y.isin(self.in_set)]
        elif self.in_ranges is not None:
            to_drop = y.copy()
            to_drop.loc[:] = False
            for in_range in self.in_ranges:
                to_drop = to_drop | (y.between(*in_range))
            post_y = post_y.loc[~to_drop]
        elif self.not_in_set is not None:
            post_y = post_y.loc[post_y.isin(self.not_in_set)]
        elif self.not_in_ranges is not None:
            to_keep = y.copy()
            to_keep.loc[:] = False
            for in_range in self.not_in_ranges:
                to_keep = to_keep | (y.between(*in_range))
            post_y = post_y.loc[to_keep]
        else:
            raise PipelineInitializationError(  # pragma: no cover
                "DropLabelsByValues: No drop conditions specified.")
        return X, post_y
