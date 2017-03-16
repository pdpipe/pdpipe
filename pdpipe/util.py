"""Utility methods for pdpipe."""


def out_of_place_col_insert(df, series, loc, column_name=None):
    """Returns a new dataframe with given column inserted at given location.

    Parameters
    ----------
    df : pandas.DataFrame
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

    Example
    -------
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
    inter_df = df.assign(**{column_name: series})
    cols = list(inter_df.columns)
    cols.insert(loc, cols.pop(cols.index(column_name)))
    return inter_df.ix[:, cols]
