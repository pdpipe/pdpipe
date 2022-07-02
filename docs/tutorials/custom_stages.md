# Creating custom pipeline stages

## Basic tranformers

Basic pipeline stages, that just want to perform the same transformation, on `fit_transform` and on `transform`, need only implement the `_transform()` and `_prec()` methods, as in this simplified version of the `ColRename` pipeline stage:

```python
import pdpipe as pdp


class ColRename(pdp.PdPipelineStage):

  def __init__(self, rename_mapper: Union[Dict, Callable], **kwargs):
    self._rename_mapper = rename_mapper
    try:
      keys_set = set(self._rename_mapper.keys())
      required_labels = list(keys_set)
      _tprec = pdp.cond.HasAllColumns(required_labels)
    except AttributeError:  # rename mapper is a callable
      _tprec = pdp.cond.AlwaysTrue()
    self._tprec = _tprec
    super_kwargs = {
      'exmsg': (
        "ColRename stage failed because not all expected columns "
        "were found in the input dataframe!
        ),
      'desc': f"Rename column with {self.mapper_repr}",
    }
    super_kwargs.update(**kwargs)
    super().__init__(**super_kwargs)

  def _prec(self, X):
    return self._tprec(X)

  def _transform(self, X, verbose):
    return X.rename(columns=self._rename_mapper)
```

??? help "What happens to y?"

    pdpipe has built-in support for X-y transformations for supervised learning, and both pipelines and pipeline stages are adaptive: If only `X`, and input dataframe, was provided, than the transformed dataframe is returned. If both `X` and `y` were returned, the appropriately transformed versions of both of them are returned, as an `(X, y)` tuple.
    However, since most pipeline stages only transform `X`, the common way to define custom pipeline stages only requires you to implement your transformation of the input dataframe. As long as you only drop and/or rearrange rows, we will make sure `y` will go through the respective transformation, as `pdpipe` makes sure `X` and `y` has an identical index.
    If you want to write pipeline stages that either add rows or change the index, you must explicitly define your transformation for both `X` and `y`. In that case, take a look at the last section, detailing how to do that.


## Column-based pipeline stages

If you wish to write a stage operating on a specific set of columns, you can extend the `ColumnsBasedPipelineStage`, which has built-in flexible interpretation abilities of the `columns` parameter, enabling you to get a single label, a list of labels, or a callable (and thus work with pdpipe's powerful `ColumnQualifier` objects), and for users to also detail column sets by exclucsion rather than inclusion.

In this case, instead of implementing the `_transform()` method, you need to implement the unique (to `ColumnsBasedPipelineStage`) version of it, `_transformation()` (and you're spared for implementint the precondition method `_prec()` yourself):

```python
import pdpipe as pdp

class DropDuplicates(pdp.ColumnsBasedPipelineStage):

  def __init__(self, **kwargs):
    super_kwargs = {
      'desc_temp': 'Drop duplicates in columns {}',  # (1)
    }
    super_kwargs.update(**kwargs)
    super_kwargs['none_columns'] = 'all'
    super().__init__(**super_kwargs)

  def _transformation(self, X, verbose, fit):
    columns = self._get_columns(X, fit=fit)
    inter_X = X.drop_duplicates(subset=columns)
    if verbose:
      print(f"{len(X) - len(inter_X)} rows dropped.")
    return inter_X
```

1. `desc_temp` is another unique constructor parameter of `ColumnsBasedPipelineStage`. You can put in a format string with `{}` as a template for the stage description, and the super-class will fill it with an appropriate string representation of the set of columns the user chose to operate on (e.g. "all", "X except for y and z", and so on).


!!! help "Getting columns when using ColumnsBasedPipelineStage"

    If you're extending `ColumnsBasedPipelineStage`, let its constructor handle all column-related parameters (see the documentation of ColumnsBasedPipelineStage) . Then, inside your implementation of the `_transformation()` method, call the `_get_columns()` method, providing it with both the input dataframe `X`, and the boolean fit context parameter `fit`. You'll get back the set of columns to operate on and ca take it from there.


## Transformers with fit status

If, alternatively, you want to build a pipeline stage that fits on the input dataframe during `fit_transform`, keeping some parameters that determine specific future transformations done using calls to `transform`, you also need to implement the `_fit_transform` method, as in this simplified version of the `Schematize` pipeline stage:

```python
import pdpipe as pdp

class Schematize(pdp.PdPipelineStage):
  """Enforces a column schema on input dataframes."""

  def __init__(
    self,
    columns: Optional[List[object]],
    **kwargs: object,
  ) -> None:
    if columns is None:
      self._adaptive = True
      self._columns = None
      self._columns_str = '<Learnable Schema>'
      exmsg = "Learnable schematize failed in precondition unexpectedly!"
    else:
      self._adaptive = False
      self._columns = _interpret_columns_param(columns)
      self._columns_str = _list_str(self._columns)
      exmsg = (
        f"Not all required columns {self._columns_str} "
        f"found in input dataframe!"
      )
    desc = (
      f"Transform input dataframes to the following schema: "
      f"{self._columns_str}"
    )
    super_kwargs = {
      'exmsg': exmsg,
      'desc': desc,
    }
    super_kwargs.update(**kwargs)
    super().__init__(**super_kwargs)

  def _prec(self, X: pandas.DataFrame) -> bool:
    if self._adaptive and not self.is_fitted:
      return True
    return set(self._columns).issubset(X.columns)

  def _transform(
      self, X: pandas.DataFrame, verbose=None) -> pandas.DataFrame:
    return X[self._columns]

  def _fit_transform(
      self, X: pandas.DataFrame, verbose=None) -> pandas.DataFrame:
    if self._adaptive:
      self._columns = X.columns
      self.is_fitted = True
      return X
    return X[self._columns]
```

!!! tip "The `is_fitted` attribute"

    Don't forget to set `self.is_fitted = True` when youre done fit-transforming your data! That's how the pipeline will know to direct future applications of the stage to the `_transform` method, where you can assume to have any fit-dependent attributes set.


## Transforming both X and y

`pdpipe` has built-in support for X-y transformations for supervised learning, and both pipelines and pipeline stages are adaptive: If only `X`, and input dataframe, was provided, than the transformed dataframe is returned. If both `X` and `y` were returned, the appropriately transformed versions of both of them are returned, as an `(X, y)` tuple.

However, since most pipeline stages only transform `X`, the common way to define custom pipeline stages only requires you to implement your transformation of the input dataframe. As long as you only drop and/or rearrange rows, we will make sure `y` will go through the respective transformation, as `pdpipe` makes sure `X` and `y` has an identical index.

If you want to write pipeline stages that either add rows or change the index, you must explicitly define your transformation for both `X` and `y`. This is done by additionally defining the `_transform_Xy()` method if you're writing a transform-only stage (with no fit/not-fit state), and the `_fit_transform_Xy()` method if you need your stage to have a fit-dependent state.
