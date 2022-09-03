
# Creating additional stages

## Extending PdPipelineStage

To use other stages than the built-in ones (see [Types of Pipeline Stages](#types-of-pipeline-stages)) you can extend the  class. The constructor must pass the `PdPipelineStage` constructor the `exmsg`, `appmsg` and `desc` keyword arguments to set the exception message, application message and description for the pipeline stage, respectively. Additionally, the `_prec` and `_transform` abstract methods must be implemented to define the precondition and the effect of the new pipeline stage, respectively.

Here is an example with a simple - non-fitable - version of the `Schematize` pipeline stage:

```python
class Schematize(PdPipelineStage):

    def __init__(self, columns: List[object],**kwargs: object) -> None:
		self._columns = columns 
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

    def _prec(self, df: pandas.DataFrame) -> bool:
        return set(self._columns).issubset(df.columns)

    def _transform(
            self, df: pandas.DataFrame, verbose=None) -> pandas.DataFrame:
        return df[self._columns]
```

Fittable custom pipeline stages should implement, additionally to the  method, the `_fit_transform` method, which should both fit pipeline stage by the input dataframe and transform transform the dataframe, while also setting `self.is_fitted = True`.

Here is the the `Schematize` stage, this time with an adaptive capability 
(activated when the parameter `columns=None`) that makes it a fittable pipeline
stage:

```python
class Schematize(PdPipelineStage):

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
            self._columns = columns 
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

    def _prec(self, df: pandas.DataFrame) -> bool:
        if self._adaptive and not self.is_fitted:
            return True
        return set(self._columns).issubset(df.columns)

    def _transform(
            self, df: pandas.DataFrame, verbose=None) -> pandas.DataFrame:
        return df[self._columns]

    def _fit_transform(
            self, df: pandas.DataFrame, verbose=None) -> pandas.DataFrame:
        if self._adaptive:
            self._columns = df.columns
            self.is_fitted = True
            return df
        return df[self._columns]
```

## Creating pipeline stages that operate on column subsets

Many pipeline stages in `pdpipe` operate on a subset of columns, allowing the
caller to determine this subset by either providing a fixed set of column
labels or by providing a callable that determines the column subset dynamically
from input dataframes. The `pdpipe.cq` module addresses a unique but important
use case of fittable column qualifier, which is to dynamically extract a column
subset on stage fit time, but keep it fixed for future transformations.

As a general rule, every pipeline stage in pdpipe that supports the `columns`
parameter should inherently support fittable column qualifier, and generally
the correct interpretation of both single and multiple labels as arguments. To
unify the implementation of such functionality, and to ease the creation of new
pipeline stages, such columns should be created by extending the
`ColumnsBasedPipelineStage` base class, found in the `pdpipe.core` module.


### Extending the `ColumnsBasedPipelineStage` class

The main way sub-classes of `ColumnBasedPipelineStage` should interact with it
is through the `columns`, `exclude_columns` and `none_columns` constructor
arguments, and the "private" `_get_columns(df, fit)` method.

Any extending subclass should accept the `columns` constructor parameter
and forward it, without transforming it, to the constructor of
`ColumnsBasedPipelineStage.` E.g.
`::: python super().__init__(columns=columns, **kwargs)`. See the
implementation of any such extending class for a more complete example.

```python
class ColDrop(ColumnsBasedPipelineStage):

    def __init__(
        self,
        columns: ColumnsParamType,
        **kwargs: object,
    ) -> None:
        self._post_cond = cond.HasNoColumn(columns)  # (1)
        super().__init__(columns=columns, **kwargs)
```

1. Unrelated to this specific use case, this is a good example of a post-condition that makes sure the output dataframe the stage returns indeed does not include the columns meant to be removed.

#### The `exclude_columns` parameter

Extending subclasses can decide if they want to expose the
`exclude_columns` parameter or not. Note that most of its functionality
can anyway be gained by providing the `columns` parameter with a column
qualifier object that is a difference between two column qualifiers; e.g.
`columns=cq.OfDtype(np.number) - cq.OfDtype(np.int64)` is equivalent to
providing `columns=cq.OfDtype(np.number),
exclude_columns=cq.OfDtype(np.int64)`. However, exposing the
`exclude_columns` parameter can allow for specific unique behaviours; for
example, if the `none_columns` parameter - which configures the behavior
when `columns` is provided with `None` - is set with
a `cq.OfDtypes('category')` column qualifier, which means that all
categorical columns are selected when `columns=None`, then exposing
`exclude_columns` allows for easy specification of the "all categorical
columns except X" by just giving a column qualifier capturing X to
`exclude_columns`, instead of having to reconstruct the default column
qualifier by hand and substract from it the one representing X.

#### Getting the columns to operate on

When wishing to get the subset of columns to operate on, in
`fit_transform` or `transform` time, it is attained by calling
`self._get_columns(df, fit=True)` (or with `fit=False` if just
transforming), providing it with the input dataframe.

#### Description and application message

Additionally, to get a description and application message with a nice
string representation of the list of columns to operate on, the
`desc_temp` constructor parameter of `ColumnsBasedPipelineStage` can be
provided with a format string with a place holder where the column list
should go. E.g. `"Drop columns {}"` for the DropCol pipeline stage.


Wrapping it all up we get the following example for the constructor of a
columns-based pipeline

!!! code-example

    ```python
	class ColDrop(ColumnsBasedPipelineStage):

		def __init__(
			self,
			columns: ColumnsParamType,
			errors: Optional[str] = None,
			**kwargs: object,
		) -> None:
			self._errors = errors
			self._post_cond = cond.HasNoColumn(columns)
			super_kwargs = {
				'columns': columns,
				'desc_temp': 'Drop columns {}',
			}
			super_kwargs.update(**kwargs)
			super_kwargs['none_columns'] = 'error'
			super().__init__(**super_kwargs)
    ```


### Fittable vs unfittable `ColumnBasedPipelineStage`

There are two correct ways to extend it, depending on whether the pipeline
stage you're creating is inherently fittable or not:

1. If the stage is NOT inherently fittable, then the ability to accept
   fittable column qualifier objects makes it so. However, to enable
   extending subclasses to implement their transformation using a single
   method, they can simply implement the abstract method
   `_transformation(self, df, verbose, fit)`. It should treat the `df` and
   `verbose` parameters normally, but forward the `fit` parameter to the
   `_get_columns` method when calling it. This is enough to get a pipeline
   stage with the desired behavior, with the super-class handling all the
   fit/transform functionality.

2. If the stage IS inherently fittable, then do not use the
   `_transformation` abstract method (it has to be implemented, so just
   have it raise a `NotImplementedError`). Instead, simply override the
   `_fit_transform` and `_transform` method of ColumnsBasedPipelineStage,
   calling the `fit` parameter of the `_get_columns` method with the
   correct arguement: `True` when fit-transforming and `False` when
   transforming.

Again, taking a look at the VERY concise implementation of simple columns-based
stages, like `ColDrop` or `ValDrop` in `pdpipe.basic_stages`, will probably make
things clearer, and you can use those implementations as a template for yours.


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

That's it!

!!! help "Getting help"

    Remember you can get help on <a href="https://gitter.im/pdpipe/community" target="_blank">our :material-wechat: Gitter chat</a> or on <a href="https://github.com/pdpipe/pdpipe/discussions" target="_blank">our :material-message-question: GitHub Discussions forum</a>.
