"""PdPipeline stages that transform the optional label column."""

from pdpipe.core import PdPipelineStage

from .exceptions import (
    PipelineInitializationError,
    UnexpectedPipelineMethodCallError,
)


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


    Example
    -------
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
        in_set=None,
        in_ranges=None,
        not_in_set=None,
        not_in_ranges=None,
        **kwargs: object,
    ):
        self.in_set = in_set
        self.in_ranges = in_ranges
        self.not_in_set = not_in_set
        self.not_in_ranges = not_in_ranges
        super_kwargs = {
            'desc': "Encode label values",
        }
        super_kwargs.update(**kwargs)
        super().__init__(**super_kwargs)

    def _prec(self, X, y):
        return y is not None

    def _transform(self, X, verbose):
        raise UnexpectedPipelineMethodCallError(
            "EncodeLabel._transform() is not expected to be called!")

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
            post_y = y.isin(self.not_in_set)
        elif self.not_in_ranges is not None:
            to_keep = y.copy()
            to_keep.loc[:] = False
            for in_range in self.not_in_ranges:
                to_keep = to_keep | (y.between(*in_range))
            post_y = post_y.loc[to_keep]
        else:
            raise PipelineInitializationError(
                "DropLabelsByValues: No drop conditions specified.")
        return X, post_y
