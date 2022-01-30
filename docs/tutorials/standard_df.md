# Standartizing Dataframes 

One of the most important roles of processing pipelines is to standartize their
output, and to make sure the assumptions made about their output by the models
consuming it are kept

`pdpipe` provide several pipeline stages that relate to this role: The
`Schematize` stage, the `ColumnDtypeEnforcer` stage and the `ConditionValidator`
stage. Let's take a look.


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

It perhaps more useful mode is the adaptive mode, activated by providing the
first parameter, `columns`, with `None`. In the adaptive mode, the stage learns
the schema of the dataframe passed through it on a `fit_transform` call, and
applies it to any future dataframe passed through it in `transform` calls.

```python
>>> import pandas as pd; import pdpipe as pdp;
>>> df1 = pd.DataFrame([[9, 4],[5, 5]], [1, 2], ['a', 'c'])
>>> df1
   a  c
1  9  4
2  5  5
>>> schematize = pdp.Schematize(None)
>>> schematize(df1)  # (1)
   a  c
1  9  4
2  5  5
>>> df2 = pd.DataFrame([[2, 4, 8],[3, 6, 9]], [1, 2], ['a', 'b', 'c'])
>>> df2
   a  b  c
1  2  4  8
2  3  6  9
>>> schematize(df2)  # (2)
   a  c
1  2  8
2  3  9
```

1. Using the stage as a callable is akin to calling `apply`, and since the
   the stage is still unfitted, `fit_transform` is callsed and the schema is
   learned. The input dataframe is thus outputted without change.

2. On the second apply call the stage is already fitted, so `transform` is
   called internally, and the input dataframe is coerced into the schema the
   stage has learned.


## Enforcing data types

The `ColumnDtypeEnforcer` stage allows us to coerce dataframe columns into a
desired datatype, with some optional powerful capabilities.

In the basic way to use the stage, we can just provide a dictionary mapping
column labels to the dtype the should be coerced into; columns not detailed by
this mapping will remain untouched:

```python
>>> import pandas as pd; import pdpipe as pdp;
>>> df = pd.DataFrame([[8,'a'],[5,'b']], [1,2], ['num', 'initial'])
>>> df
   num initial
1    8       a
2    5       b
>>> pdp.ColumnDtypeEnforcer({'num': float}).apply(df)
   num initial
1  8.0       a
2  5.0       b
```

However, the more interesting way in which `ColumnDtypeEnforcer` can be used is
by using column qualifier objects to describe critera for groups of columns to
cast to a certian dtype. This synergizes with `pdpipe`'s capability for building
pipelines that are reslient to shifts in data schema, and can be written in a 
generalizable way.

For example, here we build a dtype enforcer that will coerce into `float` any
column with a label starting with `'n'`:

```python
>>> import pandas as pd; import pdpipe as pdp;
>>> df = pd.DataFrame([[8,'a'],[5,'b']], [1,2], ['num', 'initial'])
>>> pdp.ColumnDtypeEnforcer({pdp.cq.StartWith('n'): float}).apply(df)
   num initial
1  8.0       a
2  5.0       b
```

!!! tip More column qualifiers

    Remember, column qualifiers are powerful objects, and `pdpipe` boasts
    built-in qualifiers that can help us choose columns by their data types or
    by the rate of missing values they have. See more in 
    [the section introducing column qualifiers](https://pdpipe.readthedocs.io/en/latest/starting/cq/).


## Validating conditions

The `ConditionValidator` stage allows us to to make sure various conditions
hold for input dataframes.

The most straightforward way to use it is to provide it with a function - or a
list of them - that return `True` or `False` for input dataframes:

```python
>>> import pandas as pd; import pdpipe as pdp;
>>> df = pd.DataFrame([[1,4],[4,None],[1,11]], [1,2,3], ['a','b'])
>>> df
   a     b
1  1     4
2  4  None
3  1    11
>>> validator = pdp.ConditionValidator(lambda df: len(df.columns) == 5)
>>> validator(df)
Traceback (most recent call last):
   ...
pdpipe.exceptions.FailedConditionError: ConditionValidator stage failed; some conditions did not hold for the input dataframe!
```

But again, `pdpipe` includes a special built-in type that makes this stage more
power; in this case, `Condition` objects, defined in the `pdpipe.cond` module.

For example:

```python
>>> df
   a     b
1  1     4
2  4  None
3  1    11
>>> validator = pdp.ConditionValidator(pdp.cond.HasNoMissingValues())
>>> validator(df)
Traceback (most recent call last):
   ...
pdpipe.exceptions.FailedConditionError: ConditionValidator stage failed; some conditions did not hold for the input dataframe!
```

The `cond` module includes other useful conditions, such as 
`HasAtMostMissingValues`, `HasAllColumns` and per-column conditions.
Additionally, condition objects support all boolean operators, so both
`~ cond.HasAllColumns(['a', 'b'])` and
`cond.HasAtMostMissingValues(0.1) & HasNoColumn('forbidden_column')` are valid
complex conditions that can be fed to `ConditionValidator`.

You can read more about condition objects in our Getting Started section:

[An Introduction to Conditions :fontawesome-brands-leanpub:](https://pdpipe.readthedocs.io/en/latest/starting/cond/){ .md-button .md-button--primary}


That's it!

!!! help "Getting help"

    Remember you can get help on <a href="https://gitter.im/pdpipe/community" target="_blank">our :material-wechat: Gitter chat</a> or on <a href="https://github.com/pdpipe/pdpipe/discussions" target="_blank">our :material-message-question: GitHub Discussions forum</a>.
