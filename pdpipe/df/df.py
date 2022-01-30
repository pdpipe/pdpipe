"""Handles for dynamic dataframe-method-wrapping pipeline stages."""

from pandas import DataFrame

from ..fly import (
    drop_rows_where,
    keep_rows_where,
)
from .df_transformer import (
    _DfMethodTransformerHandle,
    _is_dataframe_transform,
)
from .bound_column_potential import (
    get_bound_column_potential_by_label,
    _BoundColumnPotential,
)


class _DfHandle():

    def __init__(self) -> None:
        for attr_name in dir(DataFrame):
            attr = getattr(DataFrame, attr_name)
            if _is_dataframe_transform(attr_name, attr):
                # print(f"Adding {attr_name} of {attr}")
                handle = _DfMethodTransformerHandle(
                    method_name=attr_name,
                    doc=attr.__doc__,
                )
                setattr(self, attr_name, handle)
        self.drop_rows_where = drop_rows_where
        self.keep_rows_where = keep_rows_where

    def __getitem__(self, label: object) -> _BoundColumnPotential:
        return get_bound_column_potential_by_label(label)
