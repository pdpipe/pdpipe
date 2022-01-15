# Column Qualifiers

All `pdpipe` pipeline stages that possess the `columns` parameter can accept callables - instead of lists of labels - as valid arguments to that parameter. These callables are assumed to be column qualifiers - functions that can be applied to an input dataframe to extract the list of labels to operate on in run time.

The module `pdpipe.cq` provides a powerful class - `ColumnQualifier` - implementing this idea with various enhancements, like the ability to fit a list of labels in fit time to be retained for future transforms and support for various boolean operators between column qualifiers.

It also provides ready implementations for qualifiers qualifying columns by label, dtype and the number of missing values. This enable powerful behaviours like dropping columns by missing value frequency, scaling only integer columns or performing PCA on the subset of columns starting with the string `'tfidf_token_'`.

Read more on column qualifiers in the documentation of the `pdpipe.cq` module.
