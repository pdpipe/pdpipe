# Condition Objects

## Pdpipe pre, post and skip conditions

In `pdpipe`, pipeline stages have three optional constructor parameters that
accept callables that are treated as conditions: `prec`, `post` and `skip`.
All three assume input callables can accept a pandas.Dataframe object as input
and return either True or False. 

??? info "Pre-conditions"

    `prec` - representing the stage's precondition - determines whether a stage
    *can* be applied to an input dataframe. Accordingly, a stage throws a
    `FailedPreconditionError` if its precondition is not satisfied.

??? info "Post-conditions"

    `post` - representing the stage's postcondition - determines whether a
    stage transformed the input dataframe in the expected manner. Accordingly,
    a stage throws a `FailedPostconditionError` if its postcondition is not
    satisfied.

??? info "Skip-conditions"

    `skip` - representing the stage's skip condition - determines whether it
    *should* be applied to an input dataframe. Stage application is skipped if
    its skip-condition is satisfied


## The `pdpipe.cond` module

This module - `pdpipe.cond` - provides a way to easily generate `Condition`
objects, which are callable, and can easily be made fittable - to have their
result determined in fit time and preserved for future transforms - by
assigning the constructor parameter `fittable=True`. This enables the creation
of pipeline stages whose their effective inclusion in the pipeline is
determined only  when `fit_transform` is called; for example, whether
dimensionality reduction is required - once this decision is done in training
time it should be maintained for all future transforms of data (in test and
validation sets or in production).

## Conditions operators

Conditions objects also support the `&`, `^` and `|` binary operators -
representing boolean and, xor and or, respectively - and the `~` unary
operator - representing the boolean not operator.

So, for example, to get a condition that is satisfied by dataframes that are
missing at least one column from a list of column labels, one can use:

!!! code-example "Negation with Condition objects"

    ```python
    >>> import pandas as pd; import pdpipe as pdp;
    >>> df = pd.DataFrame(
    ...    [[8,'a',5],[5,'b',7]], [1,2], ['num', 'chr', 'nur'])
    >>> cond = ~ pdp.cond.HasAllColumns(['num', 'chr'])
    >>> cond(df)
    False
    >>> cond = ~ pdp.cond.HasAllColumns(['num','go'])
    >>> cond(df)
    True
    ```

Similarly, to get a condition that is satisfied by dataframes that both have
columns named 'foo' and 'bar' AND have no missing values, use:

!!! code-example "AND and OR with Condition objects"

    ```python
    >>> import pandas as pd; import pdpipe as pdp;
    >>> df = pd.DataFrame([[8, None],[5, 2]], [1,2], ['foo', 'bar'])
    >>> col_cond = pdp.cond.HasAllColumns(['foo', 'bar'])
    >>> missing_cond = pdp.cond.HasNoMissingValues()
    >>> (col_cond | missing_cond)(df)
    True
    >>> (col_cond & missing_cond)(df)
    False
    >>> df = pd.DataFrame([[8, 9],[5, 2]], [1,2], ['foo', 'bar'])
    >>> (col_cond & missing_cond)(df)
    True
    ```

While the same code but with XOR will yield the opposite result:

!!! code-example "XOR with Condition objects"

    ```python
    >>> import pandas as pd; import pdpipe as pdp;
    >>> df = pd.DataFrame([[8, None],[5, 2]], [1,2], ['foo', 'bar'])
    >>> col_cond = pdp.cond.HasAllColumns(['foo', 'bar'])
    >>> missing_cond = pdp.cond.HasNoMissingValues()
    >>> (col_cond ^ missing_cond)(df)
    True
    >>> df = pd.DataFrame([[8, 9],[5, 2]], [1,2], ['foo', 'bar'])
    >>> (col_cond ^ missing_cond)(df)
    False
    ```
