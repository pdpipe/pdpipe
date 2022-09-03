"""Utility methods for pdpipe."""

import numpy as np
import pandas as pd


def out_of_place_col_insert(X, series, loc, column_name=None):
    """Returns a new dataframe with given column inserted at given location.

    Parameters
    ----------
    X : pandas.DataFrame
        The dataframe into which to insert the column.
    series : pandas.Series
        The pandas series to be inserted.
    loc : int
        The location into which to insert the new column.
    column_name : str, default None
        The name to assign the new column. If None, the given series name
        attribute is attempted; if the given series is missing the name
        attribute a ValueError exception will be raised.

    Returns
    -------
    pandas.DataFrame
        The resulting dataframe.

    Examples
    --------
        >>> import pandas as pd; import pdpipe as pdp;
        >>> df = pd.DataFrame([[1, 'a'], [4, 'b']], columns=['a', 'g'])
        >>> ser = pd.Series([7, 5])
        >>> out_of_place_col_insert(df, ser, 1, 'n')
           a  n  g
        0  1  7  a
        1  4  5  b
    """
    if column_name is None:
        if series.name is None:
            raise ValueError("A column name must be supplied if the given "
                             "series is missing the name attribute.")
        column_name = series.name
    inter_X = X.assign(**{column_name: series})
    cols = list(inter_X.columns)
    cols.insert(loc, cols.pop(cols.index(column_name)))
    return inter_X.loc[:, cols]


def get_numeric_column_names(X):
    """Return the names of all columns of numeric type.

    Parameters
    ----------
    X : pandas.DataFrame
        The dataframe to get numeric column names for.

    Returns
    -------
    list of str
        The names of all columns of numeric type.

    Examples
    --------
        >>> import pandas as pd; import pdpipe as pdp;
        >>> data = [[2, 3.2, "acd"], [1, 7.2, "alk"], [8, 12.1, "alk"]]
        >>> df = pd.DataFrame(data, [1,2,3], ["rank", "ph","lbl"])
        >>> sorted(get_numeric_column_names(df))
        ['ph', 'rank']
    """
    num_cols = []
    for colbl, dtype in X.dtypes.to_dict().items():
        if np.issubdtype(dtype, np.number):
            num_cols.append(colbl)
    return num_cols


def per_column_values_sklearn_transform(X: pd.DataFrame, transform: callable):
    """Applies a 2d-array sklearn transform to 1d values arrays of a dataframe.

    Parameters
    ----------
    X : pandas.DataFrame
        The dataframe to transform.
    transform : callable
        The numpy.array-ingesting per-column transform to apply.

    Returns
    -------
    res_X : pandas.DataFrame
        The transformed dataframe.
    """
    return pd.DataFrame(
        data=np.array([
            transform(np.array([series.values]).T)[:, 0]
            for lbl, series in X.iteritems()
        ]).T,
        index=X.index,
        columns=X.columns,
    )


_LBL_PHOLDER_PREDICT = "__pdpipe_lbl_pholder_predict__"


class LabelPlaceholderForPredict(pd.Series):
    """A placeholder for y while predicting.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe to use as the index for the placeholder.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        data = np.array([_LBL_PHOLDER_PREDICT] * len(df))
        super().__init__(data, index=df.index)
