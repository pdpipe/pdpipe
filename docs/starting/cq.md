# Column Qualifiers

All `pdpipe` pipeline stages that possess the `columns` parameter can accept callables - instead of lists of labels - as valid arguments to that parameter. These callables are assumed to be column qualifiers - functions that can be applied to an input dataframe to extract the list of labels to operate on in run time.

The module `pdpipe.cq` provides a powerful class - `ColumnQualifier` - implementing this idea with various enhancements, like the ability to fit a list of labels in fit time to be retained for future transforms and support for various boolean operators between column qualifiers.

It also provides ready implementations for qualifiers qualifying columns by label, dtype and the number of missing values. This enable powerful behaviours like dropping columns by missing value frequency, scaling only integer columns or performing PCA on the subset of columns starting with the string `'tfidf_token_'`.


## The `columnns` parameter of column-based stages

Most pipeline stages in pdpipe can accept three types of variables as arguments
for the `columns` parameter of their constructor: a single column label, a list
of column labels, or a callable. The first is interpreted as the label of the
single column on which the pipeline stage should operate and the second as a
list of such labels, while the a callable is assumed to determine dynamically
what columns should the stage be applied to. It is thus supplied with the
entire input dataframe, and is expected to return a list of column labels. This
is true for every application of the pipeline stage, both in fit time and in
any future transform.

A naive callable, such as `lambda df: [lbl for lbl in df.columns if lbl[0] ==
'a']`, meant to cause the pipeline stage to operate on any column with a label
starting with the letter 'a', might result in unexpected errors: if after the
pipeline was fitted it gets ― in transform time ― a dataframe with new columns
starting with 'a', it is will transform them as well, which will in turn might
(1) lead to unexpected errors, as the newer columns might not be valid input to
that stage, and (2) lead to a change in schema, which might cause errors down
the pipeline, especially if there's a fitted machine learning model down the
pipeline.

Of course, this might be the desired behaviour ― to transform columns 'alec'
and 'alex' on the first `apply` call and 'alec' and 'apoxy' in transform time ―
but usually this is not the case. In fact, in common machine learning
scenarios ― whether it is fitting pre-processing parameters on the train set
and using the resulting pipeline to transform the test and validation sets, or
fitting the pre-processing pipeline on current data and deploying it to
transform incoming data in production ― we would exepct this criterion to be
applied once, when the pipeline is being fit, and for future calls for it to
only transform the 'alec' and 'alex' columns, ignoring any other columns
starting with 'a' that are newly-encountered in transform time, and also
explicitly fails if either once of the two columns, 'alec' and 'alex', is
missing. This kind of behaviour is the only way to ensure preservation of both
form and semantics of input vectors to our models down the pipeline.


## ColumnQualifier objects

To enable this more sophisticated behaviour, the `pdpipe.cq` module exposes a
way to easily generate `ColumnQualifier` objects, which are callables that do
exactly what was described above: Apply some criteria to determine a set of
input columns when a pipeline is being fitted, but fixing it afterwards, on
future calls.

Practically, this objects all expose `fit`, `transform` and `fit_transform`
methods, and while the first time they are called the `fit_transform` method is
called, future calls will actually call the `transform` method. Also, since
they already expose this more powerful API, pipeline stages use it to enable
an even more powerful (and quite frankly, expected) behavior: When a pipeline's
`fit_transfrom` or `fit` methods are called, it calls the `fit_transform`
method of the column qualifier it uses, so the qualifier itself is refitted
even if it is already fit. Naturally, if the callable has no `fit_transform`
method the code gracefully backs-off to just applying it, which allows the use
of unfittable functions and lambdas as column qualifiers as well.


## Wrapping callables as ColumnQualifiers

Note that any callable can be wrapped in a ColumnQualifier object to achieve
this fittable behaviour. For example, to get a pipeline stage that drops all
columns of data type `numpy.int64`, but also "remembers" that list after fit:

```python
pipeline += pdp.ColDrop(columns=pdp.cq.ColumnQualifier(lambda df: [
  l for l, s in df.iteritems()
  if s.dtype == np.int64
]))
```

## Column qualifier operators

ColumnQualifier objects also support the &, ^ and | binary operators -
representing boolean and, xor and or, respectively - and the ~ unary boolean
operator - representing the boolean not operator. Finally, the - binary
operator is implemented to represent the NOT IN non-symetric binary relation
between two qualifiers (the difference operator on the resulting sets).

So for example, to get a qualifier that qualifies all columns that have AT
LEAST two missing values, one can use:


```python
>>> import pandas as pd; import pdpipe as pdp;
>>> df = pd.DataFrame(
...    [[None, 1, 2],[None, None, 5]], [1,2], ['ph', 'grade', 'age'])
>>> cq = ~ pdp.cq.WithAtMostMissingValues(1)
>>> cq(df)
['ph']
```

While to get a qualifier matching all columns with at most one missing value
AND starting with 'gr' one can use:

```python
>>> import pandas as pd; import pdpipe as pdp;
>>> df = pd.DataFrame(
...    [[None, 1, 2],[None, None, 5]], [1,2], ['grep', 'grade', 'age'])
>>> cq = pdp.cq.WithAtMostMissingValues(1) & pdp.cq.StartWith('gr')
>>> cq(df)
['grade']
```

And a qualifier that qualifies all columns with no missing values except those
that start with 'b' can be generated with:

```python
>>> import pandas as pd; import pdpipe as pdp;
>>> df = pd.DataFrame(
...    [[1, 2, 3, 4],[5, 6, 7, None]], [1,2], ['abe', 'bee', 'cry', 'no'])
>>> cq = pdp.cq.WithoutMissingValues() - pdp.cq.StartWith('b')
>>> cq(df)
['abe', 'cry']
```
