# Building Pipelines: Generator functions VS constructors

This is a short one, showcasing two ways to dynamically build complex,
parameterized, pipelines.

## Generator functions for pipelines

One way to generate complex pipeline is by defining a generator function whose
signature is composed of both all the parameters that define the structure of
the pipeline (like whether or not to include a specific pipeline stage, or 
a whole section of the pipeline), AND all the parameters inner pipeline stages
need to get when initializing them.

```python
from typing import List

import pandas as pd
import pdpipe as pdp


class Halfer:

  def __init__(self, columns_to_halve: List[object]) -> None:
    self.columns_to_halve = columns_to_halve

  def __call__(self, row: pd.Series) -> pd.Series:
    new = {
      f'{lbl}/2': row[lbl] / 2
      for lbl in self.columns_to_halve
    }
    return pd.Series(new)


def pipeline_constructor(
  columns_to_drop: List[object],
  columns_to_half: List[object],
  scale: bool,
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
  scale : bool
    If True, the last pipeline stage min-max scales all numerical columns.
    Otherwise, no such pipeline stage is appended to the pipeline.

  Returns
  -------
  pipeline : pdpipe.PdPipeline
    The resulting pipeline constructed by this constructor.
  """
  stages = [
    pdp.ColDrop(columns_to_drop),
    pdp.ApplyToRows(
      func=Halfer(columns_to_half),
      follow_column='years',
    ),
  ]
  if scale:
    stages.append(pdp.Scale('MinMaxScaler'))
  return pdp.PdPipeline(stages)
```


## Constructors for pipelines


Another option to achieve the same result is to directly extend the `pdpipe.PdPipeline` class. The created stages can be sent to the constructor of the super class, as can any extra keyword-arguments, allowing you to preserve all functionality of the `PdPipeline` class.


```python
from typing import List

import pandas as pd
import pdpipe as pdp


class Halfer:

  def __init__(self, columns_to_halve: List[object]) -> None:
    self.columns_to_halve = columns_to_halve

  def __call__(self, row: pd.Series) -> pd.Series:
    new = {
      f'{lbl}/2': row[lbl] / 2
      for lbl in self.columns_to_halve
    }
    return pd.Series(new)


class MyPipeline(pdp.PdPipeline):

  def __init__(
    columns_to_drop: List[object],
    columns_to_half: List[object],
    scale: bool,
    **kwargs: object,  # (1)
  ) -> None:
    """My pandas dataframe-processing pipeline, according to some input arguments.

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
    scale : bool
      If True, the last pipeline stage min-max scales all numerical columns.
      Otherwise, no such pipeline stage is appended to the pipeline.
    """
    stages = [
      pdp.ColDrop(columns_to_drop),
      pdp.ApplyToRows(
        func=Halfer(columns_to_half),
        follow_column='years',
      ),
    ]
    if scale:
      stages.append(pdp.Scale('MinMaxScaler'))
    super().__init__(stages=stages, **kwargs)
```

1. This is the correct way to type-hint the ``**kwargs`` variable argument
   operator. We only need to hint the type of values in the ``kwargs`` dict,
   and thus, if we don't want to contrain them at all, we type-hint ``object``.

That's it!

!!! help "Getting help"

    Remember you can get help on <a href="https://gitter.im/pdpipe/community" target="_blank">our :material-wechat: Gitter chat</a> or on <a href="https://github.com/pdpipe/pdpipe/discussions" target="_blank">our :material-message-question: GitHub Discussions forum</a>.
