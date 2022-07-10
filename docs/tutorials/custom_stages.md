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

Take, for example, a very simplified version of the `DropLabelsByValues` stage (the actual version supports several ways to detail the by-value dropping logic), as an example for a transform-only X-y tranformer:


```python
class DropLabelsByValues(PdPipelineStage):

    def __init__(
        self,
        in_set: Optional[Iterable[object]] = None,
        **kwargs: object,
    ) -> None:
        self.in_set = in_set
        super_kwargs = {
            'desc': "Drop labels by values",
        }
        super_kwargs.update(**kwargs)
        super().__init__(**super_kwargs)

    def _prec(self, X, y):  # (1)
        return y is not None

    def _transform(self, X, verbose):  # (2)
        raise UnexpectedPipelineMethodCallError(  # (3)
            "DropLabelsByValues._transform() is not expected to be called!")

    def _transform_Xy(self, X, y, verbose):  # (4)
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
        return X, post_y  # (5)

```

1. We implement a standard precondition for pipeline stages that wish to transform `y`, or both `X` and `y`; checking that the input `y` parameter isn't `None`.
2. We have to implement `_transform()` as its an abstract method of `PdPipelineStage`.
3. We make sure our benign implementation of `_transform()` raise the unique `UnexpectedPipelineMethodCallError` exception on each call. This code would never be called (unless someone calls it by hand, or an implementation bug is found in the `pdpipe` library itself.
4. Unlike `_transform()`, the `_transform_Xy()` recieves both `X` and `y` as parameters, and return both of them.
5. A nice thing that `PdPipelineStage` does for us is automatically re-align and re-index `X` according to the transformed `y` (and the other way around), so the method just needs to detail the transformation for `y`. You may, of course, transform both, or manually re-align them using `return X.loc[post_y.index], post_y`.

Similarly, the `EncodeLabel` pipeline stage provides a simple example for an X-y tranformer with a fit-state, so one implementing both the `_transform_Xy()` and the `_fit_transform_Xy()` methods: 

```python
class EncodeLabel(PdPipelineStage):

    def __init__(self, **kwargs: object) -> None:
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

    def _fit_transform_Xy(self, X, y, verbose):
        self.encoder_ = sklearn.preprocessing.LabelEncoder()
        post_y = self.encoder_.fit_transform(y)
        post_y = pd.Series(data=post_y, index=y.index)
        self.is_fitted = True
        return X, post_y

    def _transform_Xy(self, X, y, verbose):
        try:
            post_y = self.encoder_.transform(y)
            post_y = pd.Series(data=post_y, index=y.index)
            return X, post_y
        except AttributeError:
            raise UnfittedPipelineStageError("EncodeLabel is not fitted!")
```


## Continue to the in-depth guide

A more in-depth guide to subclassing `pdpipe.PdPipelineStage`, and related classes, can be found in our Develop section:

[Creating Additional Stages :fontawesome-brands-leanpub:](https://pdpipe.readthedocs.io/en/latest/develop/custom/){ .md-button .md-button--primary}

That's it!

!!! help "Getting help"

    Remember you can get help on <a href="https://gitter.im/pdpipe/community" target="_blank">our :material-wechat: Gitter chat</a> or on <a href="https://github.com/pdpipe/pdpipe/discussions" target="_blank">our :material-message-question: GitHub Discussions forum</a>.
