# Types of Pipeline Stages

All built-in stages are thoroughly documented, including examples; if you find any documentation lacking please open an issue. A list of briefly described available built-in stages follows:

## Built-in pandas methods

Ad-hoc pipeline stages that wrap any `pandas.DataFrame` built-in method that returns a dataframe object can be easily created using the `pdpipe.df` submodule:

For example `pdp.df.dropna(axis=1)` will return a `pdpipe.PdPipelineStage`
object that will call the `dropna` method of input DataFrames with the `axis=1`
keyword argument provided, and return the resulting dataframe object
(practically dropping any column with a missing value from the input
dataframe).

```python title="Using pdp.df"
pipeline = pdp.PdPipeline([
    pdp.df.set_index(keys='datetime'),  # (1)
    pdp.ColDrop('age'),
])
```

1. `keys` is simply a keyword argument of `pandas.DataFrame.set_index`!

!!! info

    `pdpipe` pipeline stages never alter input dataframes, so the `inplace` keyword argument is always ignored, even if provided.


!!! attention

    All method parameters are fixed on pipeline stage creation time, and must be explicitly provided as keyword arguments, and not as positional ones.


## Basic Stages

Refer to submodule `pdpipe.basic_stages`

* AdHocStage - Define custom pipeline stages on the fly.
* ColDrop - Drop columns by name.
* ValDrop - Drop rows by by their value in specific or all columns.
* ValKeep - Keep rows by by their value in specific or all columns.
* ColRename - Rename columns.
* DropNa - Drop null values. Supports all parameter supported by pandas.dropna function. 
* FreqDrop - Drop rows by value frequency threshold on a specific column.
* ColReorder - Reorder columns.
* RowDrop - Drop rows by callable conditions.
* Schematize - Learn a dataframe schema on fit and transform to it on future transforms.
* DropDuplicates - Drop duplicate values in a subset of columns.

## Column Generation

Refer to submodule `pdpipe.col_generation`

* Bin - Convert a continuous valued column to categoric data using binning.
* OneHotEncode - Convert a categorical column to the several binary columns corresponding to it.
* MapColVals - Replace column values by a map.
* ApplyToRows - Generate columns by applying a function to each row.
* ApplyByCols - Generate columns by applying an element-wise function to columns.
* ColByFrameFunc - Add a column by applying a dataframe-wide function.
* AggByCols - Generate columns by applying an series-wise function to columns.
* Log - Log-transform numeric data, possibly shifting data before.

## Text Stages

Refer to submodule `pdpipe.text_stages`

* RegexReplace - Replace regex occurences in columns of strings.
* DropTokensByLength - Drop tokens in token lists by token length.
* DropTokensByList - Drop every occurence of a given set of string tokens in token lists.

## Scikit-learn-dependent Stages

Refer to submodule `pdpipe.sklearn_stages`

* Encode - Encode a categorical column to corresponding number values.
* Scale - Scale data with any of the sklearn scalers.
* TfidfVectorizeTokenLists - Transform a column of token lists into the correponding set of tfidf vector columns.

## nltk-dependent Stages

Refer to submodule `pdpipe.nltk_stages`

* TokenizeWords - Tokenize a sentence into a list of tokens by whitespaces.
* UntokenizeWords - Joins token lists into whitespace-seperated strings.
* RemoveStopwords - Remove stopwords from a tokenized list.
* SnowballStem - Stems tokens in a list using the Snowball stemmer.
* DropRareTokens - Drop rare tokens from token lists.
