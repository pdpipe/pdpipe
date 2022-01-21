# Fly Handles

`pdpipe` provides a few unique handles that allow the creation of pipeline
stages in a very intuitive syntax.

You can use `drop_rows_where` and `keep_rows_where` as handles to the future dataframe,
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
