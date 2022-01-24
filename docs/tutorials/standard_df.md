# Standartizing Dataframes 

One of the most important roles of processing pipelines is to standartize their
output, and to make sure the assumptions made about their output by the models
consuming it are kept

`pdpipe` provide several pipeline stages that relate to this role; let's take
a look.


## Schematization

The `Schematize` pipeline stage provides a concise way to align any dataframe
passing through it to a specific column schema.

```python
>>> import pandas as pd; import pdpipe as pdp;
>>> df = pd.DataFrame([[2, 4, 8],[3, 6, 9]], [1, 2], ['a', 'b', 'c'])
>>> pdp.Schematize(['a', 'c']).apply(df)
   a  c
1  2  8
2  3  9
>>> pdp.Schematize(['c', 'b']).apply(df)
   c  b
1  8  4
2  9  6
```

## Enforcing data types

## Validating conditions


That's it!

!!! help "Getting help"

    Remember you can get help on <a href="https://gitter.im/pdpipe/community" target="_blank">our :material-wechat: Gitter chat</a> or on <a href="https://github.com/pdpipe/pdpipe/discussions" target="_blank">our :material-message-question: GitHub Discussions forum</a>.
