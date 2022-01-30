# Fly Handles

## The df handle

`pdpipe`'s `df` handle provides a couple of unique ways to create pipeliene stages. For ease of use, you can import it the following manner:

```python
from pdpipe import df
```

The first is the ability to create stages that applies any `pandas.DataFrame` method that outputs a dataframe to input dataframes, such as `set_index`, `fill_na`, `rename`, etc., for example:

```python
pline = pdp.PdPipeline([
    df.set_index('id'),
    df.fillna(value=3, method='ffill'),
])
```


The second ability is the creation of column assignment stages in an intuitive manned - using the `<<` operator to denote assignment - allowing the user of operators between columns, Series objects and scalars, and the use of `pandas.Series` methods:

```python
pline = pdp.PdPipeline([
    df['n'] << df['a'] & ~df['b'],
    df['g'] << df['c'] + (3 * df['d']) + 5,
    df['j'] << df['s'].map({1: 2, 2: 8}) + pd.Series([1, 2, 3, 4]),
])
```


## Addtional fly handles

You can also use then `drop_rows_where` and `keep_rows_where` as handles to the future dataframe,
using the `[]` indexing syntax to select a (single) column to apply the logic
by, and regular binary operators such as >, >=, ==, !=, etc. to express the
condition by which rows will be kept or dropped.

```python
>>> import pandas as pd; import pdpipe as pdp;
>>> df = pd.DataFrame([[1,4],[4,5],[5,11]], [1,2,3], ['a','b'])
>>> df
   a   b
1  1   4
2  4   5
3  5  11
>>> pipeline = pdp.PdPipeline([
...     pdp.drop_rows_where['a'] > 4,
... ])
>>> pipeline(df)
   a   b
1  1   4
2  4   5
```

Additionally, the resulting stages can be naturaly combined by logical binary operators:
& for AND, | for OR and ^ for XOR, and can also be inverted with the `~`
operator.

```python
>>> import pandas as pd; import pdpipe as pdp;
>>> pipeline = pdp.PdPipeline([
...     ~ (pdp.drop_rows_where['a'] > 4),
... ])
>>> pipeline(df)
   a   b
3  5  11
>>> pipeline = pdp.PdPipeline([
...     (pdp.drop_rows_where['a'] > 3) & (pdp.drop_rows_where['b'] < 10),
... ])
>>> pipeline(df)
   a   b
1  1   4
3  5  11
```
