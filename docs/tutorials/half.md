# Halving Columns: A Demonstration of Column Transformations in `pdpipe`

Let's say you have a dataframe with numerical columns, and you want to generate
new columns that hold the halved values of some of the original columns. How 
would you go about it?

That depends on *when* do you know which columns you want to half. Let's go
over the different scenarios, which will demonstrate increasingly advanced
use cases of `pdpipe`.

## 1. Pre-determined set of columns

If you always know exactly which columns are those beforehand - down to their
exact labels - when constructing the pipelines, they are not parameters or
hyperparameters of our pipeline. They should be hardcoded, for example, in the
following way:

```python
_COLUMNS_TO_HALVE = ['year', 'revenue']

def halfer(row):
  new = {
    f'{lbl}/2': row[lbl] / 2
    for lbl in _COLUMNS_TO_HALVE
  }
  return pd.Series(new)

COL_HALVER = pdp.ApplyToRows(halfer, follow_column='years')
```

So here we've used a dict comprehension to create a new half-column for each column in a list of pre-determined columns we know. This will always operate on the same set of columns, regardless of the input dataframe (and it will fail if not all of them are contained in it).

I've also put everything in the global scope of the imaginary Python script file we're writing. If this is in a notebook, it probably looks the same, possibly minus the all-caps to signify global variables.


## 2. Columns are known on pipeline creation time

If this is not set in stone, but is indeed always known on pipeline creation time (but may change between different uses of the same pipeline, or perhaps pipeline "template), then you need a constructor function to construct the pipeline stage on pipeline creation, which means you just probably want a pipeline constructor function. Then, `year` and `revenue` are parameters of the constructor, and not of the pipeline stage or the function themselves. 

```python
from typing import List

import pdpipe as pdp

def _halfer_constructor(columns_to_halve: List[object]) -> callable:

  # having this defined as a named function and not a lambda makes the resulting
  # pipeline stage, and thus the whole pipeline, pickle-able/serializable
  def halfer(row):
    new = {
      f'{lbl}/2': row[lbl] / 2
      for lbl in columns_to_halve
    }
    return pd.Series(new)
  return halfer


def pipeline_constructor(
  columns_to_drop: List[object],
  columns_to_half: List[object],
) -> pdp.PdPipeline:
  """Constructs my pandas dataframe-processing pipeline, according to some input arguments.

  Parameters
  ----------
  columns_to_drop : list of objects
     A list of the labels of the columns to drop.
     Any Python object that can be used as pandas label can be included in the list.
  columns_to_half : list of objects
     A list of the labels of the columns to half.
     For each such a column, an additional new column, containing its halved values, is generated.
     Each new column has the label "x/2", where "x" is the label of the corresponding original column.
     Any Python object that can be used as pandas label can be included in the list.

  Returns
  -------
  pipeline : pdpipe.PdPipeline
    The resulting pipeline constructed by this constructor.
  """
  return pdp.PdPipeline([
    pdp.ColDrop(columns_to_drop),
    pdp.ApplyToRows(
      func=_halfer_constructor(columns_to_half),
      follow_column='years',
    ),
  ])
```


## 3. Columns are determined on pipeline fit

In this scenario, you don't know beforehand the exact labels of the columns you want to half, but let's assume you know something about them. Perhaps you want to half all float-valued columns, or all columns with labels starting with the word "revenue", etc.

Luckily, `pdpipe` actually contains a strong mechanism to provide such functionality , called column qualifiers. You can read more about them on the [column qualifiers introduction page](https://pdpipe.readthedocs.io/en/latest/starting/cq/).

We will also have to switch to a little more powerful and specific pipeline stage, the `MapColVals` stage ([see doc here](https://pdpipe.readthedocs.io/en/latest/reference/col_generation/#pdpipe.col_generation.MapColVals)). Basically, we're going to provide a mapping function mapping each value if the old column to a new value in the generated column (in our case, the divide-by-2 function), and we are going to provide - instead of a list of columns - an object called a column qualifier which will determine on application time the list of column the stage should be applied to, using some sort of criteria.

If, for example, we want to generate new, half-value, columns for each column with float values in it, we can do so like this:

```python
import numpy as np
import pdpipe as pdp

float_col_halver = pdp.MapColVals(
  columns=pdp.cq.OfDtypes(np.float),
  value_map=lambda x: x/2,
  drop=False,
  suffix='_half',
)
```

This neat little pipeline stage will, when a dataframe is first passed through it, build a list of all columns of dtype float (any kind of numpy float, be it `float32`, `float64` and so on), and save it. Then, for each such column, it will apply the value map element-wise to generate a new `pandas.Series`, which it will assign to the input dataframe under the label `'x_half`', where `x` is the label of the original column.

The cool thing is, that if applied once on a dataframe — let's say, your training set — it will remember the list of columns it "chose" by the criteria you fed it, and will only apply it to the same list of columns on any future dataframe, even if it has additional float columns. This property is invaluable in ML scenarios, when you need to generate a fixed schema for the model who follows. You can't just half a new column on inference time just because something changed in the input data (you actually have to discard it).

!!! tip "Tip: Advanced column qualifiers"

    Now, if you instead want to halve all columns with string labels starting with "revenue", you could use `pdp.cq.StartWith("revenue")` instead. If you want all number columns (int or float or others), you could use `pdp.cq.OfNumericDtypes()`. And the coolest thing? You can easily combine such criteria:

    ```#!python pdp.cq.WithAtMostMissingValues(1) & pdp.cq.StartWith('revenue')``` will make sure the stage is applied only to columns with at most one missing value and a label. ```#!python pdp.cq.WithoutMissingValues() - pdp.cq.StartWith('b')``` is a qualifier that qualifies all columns with no missing values except those that start with 'b'. And ```#!python pdp.cq.StartWith('revenue') | pdp.cq.StartWith('expenses')``` will yield all columns that start with either "expenses" or "revenue". You can also create custom conditions with ```#!python pdp.cq.ByColumnCondition(some_function)```.

??? help "How to keep things pickle-able?"

    If you want the whole thing to be pickle-able, the callable you provide the `value_map` parameters needs to be a named function rather than a `lambda`.

??? help "How to drop the source columns?"

    If you want to drop the original columns, just provide the constructor
    with `drop=True`.


## 4. Columns are determined on each application

Ok, say all of that sounds great, but you're not in the specific fit-vs-transform scenario that is common in ML. You just want to build a pipeline which includes a stage that halves all revenue columns in an input dataframe, and you don't care if it's a different list every time. No problem.

Column qualifiers have the `fittable` constructor keyword argument. Simply set it to `False` and they will filter columns from input dataframes on each application, and will not "learn" to output a specific set after the first application:

```python
import numpy as np
import pdpipe as pdp

float_col_halver = pdp.MapColVals(
  columns=pdp.cq.OfDtypes(np.float, fittable=False),
  value_map=lambda x: x/2,
  drop=False,
  suffix='_half',
)
```

That's it!

!!! help "Getting help"

    Remember you can get help on <a href="https://gitter.im/pdpipe/community" target="_blank">our :material-wechat: Gitter chat</a> or on <a href="https://github.com/pdpipe/pdpipe/discussions" target="_blank">our :material-message-question: GitHub Discussions forum</a>.
