
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

!!! code-example annotate "Forwarding the `columns` parameter"

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
