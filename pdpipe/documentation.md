

Features
========

* A simple interface.
* Informative prints and errors on pipeline application.
* Chaining pipeline stages constructor calls for easy, one-liners pipelines.
* Pipeline arithmetics.
* Easier handling of mixed data (numeric, categorical and others).
* [Fully tested on Linux, macOS and Windows systems](https://travis-ci.org/pdpipe/pdpipe).
* Compatible with Python 3.5+.
* Pure Python.


Design Decisions
----------------

* **Extra informative naming:** Meant to make pipelines very readable, understanding their entire flow by pipeline stages names; e.g. ColDrop vs. ValDrop instead of an all-encompassing Drop stage emulating the `pandas.DataFrame.drop` method.
* **Data science-oriented naming** (rather than statistics).
* **A functional approach:** Pipelines never change input DataFrames. Nothing is done "in place".
* **Opinionated operations:** Help novices avoid mistake by default appliance of good practices; e.g., one-hot-encoding (creating dummy variables) a column will drop one of the resulting columns by default, to avoid [the dummy variable trap](http://www.algosome.com/articles/dummy-variable-trap-regression.html) (perfect [multicollinearity](https://en.wikipedia.org/wiki/Multicollinearity)).
* **Machine learning-oriented:** The target use case is transforming tabular data into a vectorized dataset on which a machine learning model will be trained; e.g., column transformations will drop the source columns to avoid strong linear dependence.


# Installation

Install `pdpipe` with:

```python
  pip install pdpipe
```

Some pipeline stages require `scikit-learn`; they will simply not be loaded if `scikit-learn` is not found on the system, and `pdpipe` will issue a warning. To use them you must also [install scikit-learn](http://scikit-learn.org/stable/install.html).


Similarly, some pipeline stages require `nltk`; they will not be loaded if `nltk` is not found on your system, and `pdpipe` will issue a warning. To use them you must additionally [install nltk](http://www.nltk.org/install.html).


Basic Use
=========

The awesome Tirthajyoti Sarkar wrote [an excellent practical introduction on how to use pdpipe](https://tirthajyoti.github.io/Notebooks/Pandas-pipeline-with-pdpipe). Read it now [on his website](https://tirthajyoti.github.io/Notebooks/Pandas-pipeline-with-pdpipe)!

For a thorough overview of all the capabilities of `pdpipe`, continue below:

Pipeline Stages
---------------

### Creating Pipeline Stages

You can create stages with the following syntax:

```python
  import pdpipe as pdp
  drop_name = pdp.ColDrop("Name")
```

All pipeline stages have a predefined precondition function that returns True for dataframes to which the stage can be applied. By default, pipeline stages raise an exception if a DataFrame not meeting their precondition is piped through. This behaviour can be set per-stage by assigning `exraise` with a bool in the constructor call. If `exraise` is set to `False` the input DataFrame is instead returned without change:

```python
  drop_name = pdp.ColDrop("Name", exraise=False)
```


### Applying Pipeline Stages

You can apply a pipeline stage to a DataFrame using its `apply` method:

```python
  res_df = pdp.ColDrop("Name").apply(df)
```

Pipeline stages are also callables, making the following syntax equivalent:

```python
  drop_name = pdp.ColDrop("Name")
  res_df = drop_name(df)
```

The initialized exception behaviour of a pipeline stage can be overridden on a per-application basis:

```python
  drop_name = pdp.ColDrop("Name", exraise=False)
  res_df = drop_name(df, exraise=True)
```

Additionally, to have an explanation message print after the precondition is checked but before the application of the pipeline stage, pass `verbose=True`:

```python
  res_df = drop_name(df, verbose=True)
```

All pipeline stages also adhere to the `scikit-learn` transformer API, and so have `fit_transform` and `transform` methods; these behave exactly like `apply`, and accept the input dataframe as parameter `X`. For the same reason, pipeline stages also have a `fit` method, which applies them but returns the input dataframe unchanged.


### Fittable Pipeline Stages

Some pipeline stages can be fitted, meaning that some transformation parameters are set the first time a dataframe is piped through the stage, while later applications of the stage use these now-set parameters without changing them; the `Encode` scikit-learn-dependent stage is a good example.

For these type of stages the first call to `apply` will both fit the stage and transform the input dataframe, while subsequent calls to `apply` will transform input dataframes according to the already-fitted transformation parameters.

Additionally, for fittable stages the `scikit-learn` transformer API methods behave as expected:

* `fit` sets the transformation parameters of the stage but returns the input dataframe unchanged.
* `fit_transform` both sets the transformation parameters of the stage and returns the input dataframe after transformation.
* `transform` transforms input dataframes according to already-fitted transformation parameters; if the stage is not fitted, an `UnfittedPipelineStageError` is raised.

Again, `apply`, `fit_transform` and  are all of equivalent for non-fittable pipeline stages. And in all cases the `y` parameter of these methods is ignored.


Pipelines
---------

### Creating Pipelines

Pipelines can be created by supplying a list of pipeline stages:

```python
    pipeline = pdp.PdPipeline([pdp.ColDrop("Name"), pdp.OneHotEncode("Label")]
```

Additionally, the  method can be used to give stages as positional arguments.

```python
    pipeline = pdp.make_pdpipeline(pdp.ColDrop("Name"), pdp.OneHotEncode("Label"))
```


### Printing Pipelines

A pipeline structure can be clearly displayed by printing the object:

```python
  >>> drop_name = pdp.ColDrop("Name")
  >>> binar_label = pdp.OneHotEncode("Label")
  >>> map_job = pdp.MapColVals("Job", {"Part": True, "Full":True, "No": False})
  >>> pipeline = pdp.PdPipeline([drop_name, binar_label, map_job])
  >>> print(pipeline)
  A pdpipe pipeline:
  [ 0]  Drop column Name
  [ 1]  OneHotEncode Label
  [ 2]  Map values of column Job with {'Part': True, 'Full': True, 'No': False}.
```

### Pipeline Arithmetics

Alternatively, you can create pipelines by adding pipeline stages together:

```python
  pipeline = pdp.ColDrop("Name") + pdp.OneHotEncode("Label")
```

Or even by adding pipelines together or pipelines to pipeline stages:

```python
  pipeline = pdp.ColDrop("Name") + pdp.OneHotEncode("Label")
  pipeline += pdp.MapColVals("Job", {"Part": True, "Full":True, "No": False})
  pipeline += pdp.PdPipeline([pdp.ColRename({"Job": "Employed"})])
```

### Pipeline Chaining

Pipeline stages can also be chained to other stages to create pipelines:

```python
  pipeline = pdp.ColDrop("Name").OneHotEncode("Label").ValDrop([-1], "Children")
```

### Pipeline Slicing

Pipelines are Python Sequence objects, and as such can be sliced using Python's slicing notation, just like lists:

```python
  >>> pipeline = pdp.ColDrop("Name").OneHotEncode("Label").ValDrop([-1], "Children").ApplyByCols("height", math.ceil)
  >>> pipeline[0]
  Drop column Name
  >>> pipeline[1:2]
  A pdpipe pipeline:
  [ 0] OneHotEncode Label
```
Pipelines can also be sliced by the stages `name` parameter, notice when running `pipeline[['name1', 'name2']]` a new pipeline will returned with all stages that they `name` is 'name1' or 'name2', and when running `pipeline['name'] only the first stage that has the 'name' will return.:

```python
  >>> pipeline = pdp.ColDrop("Name", name="dropName").OneHotEncode("Label", name="encoder").ValDrop([-1], "Children").ApplyByCols("height", math.ceil)
  >>> pipeline['dropName']
  PdPipelineStage: Drop columns Name
  >>> pipeline[['dropName', 'encoder']]
  A pdpipe pipeline:
  [ 0]  Drop columns Name
  [ 1]  One-hot encode Label
```

### Applying Pipelines

Pipelines are pipeline stages themselves, and can be applied to a DataFrame using the same syntax, applying each of the stages making them up, in order:

```python
  pipeline = pdp.ColDrop("Name") + pdp.OneHotEncode("Label")
  res_df = pipeline(df)
```

Assigning the  parameter to a pipeline apply call with a bool sets or unsets exception raising on failed preconditions for all contained stages:

```python
  pipeline = pdp.ColDrop("Name") + pdp.OneHotEncode("Label")
  res_df = pipeline.apply(df, exraise=False)
```

Additionally, passing ``verbose=True`` to a pipeline apply call will apply all pipeline stages verbosely:

```python
  res_df = pipeline.apply(df, verbose=True)
```

Finally, `fit`, `transform` and `fit_transform` all call the corresponding pipeline stage methods of all stages composing the pipeline.


Column Qualifiers
-----------------

All `pdpipe` pipeline stages that possess the `columns` parameter can accept callables - instead of lists of labels - as valid arguments to that parameter. These callables are assumed to be column qualifiers - functions that can be applied to an input dataframe to extract the list of labels to operate on in run time.

The module `pdpipe.cq` provides a powerful class - `ColumnQualifier` - implementing this idea with various enhancements, like the ability to fit a list of labels in fit time to be retained for future transforms and support for various boolean operators between column qualifiers.

It also provides ready implementations for qualifiers qualifying columns by label, dtype and the number of missing values. This enable powerful behaviours like dropping columns by missing value frequency, scaling only integer columns or performing PCA on the subset of columns starting with the string `'tfidf_token_'`.

Read more on column qualifiers in the documentation of the `pdpipe.cq` module.



Types of Pipeline Stages
========================

All built-in stages are thoroughly documented, including examples; if you find any documentation lacking please open an issue. A list of briefly described available built-in stages follows:

Basic Stages
------------

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

Column Generation
-----------------

Refer to submodule `pdpipe.col_generation`

* Bin - Convert a continuous valued column to categoric data using binning.
* OneHotEncode - Convert a categorical column to the several binary columns corresponding to it.
* MapColVals - Replace column values by a map.
* ApplyToRows - Generate columns by applying a function to each row.
* ApplyByCols - Generate columns by applying an element-wise function to columns.
* ColByFrameFunc - Add a column by applying a dataframe-wide function.
* AggByCols - Generate columns by applying an series-wise function to columns.
* Log - Log-transform numeric data, possibly shifting data before.

Text Stages
-----------

Refer to submodule `pdpipe.text_stages`

* RegexReplace - Replace regex occurences in columns of strings.
* DropTokensByLength - Drop tokens in token lists by token length.
* DropTokensByList - Drop every occurence of a given set of string tokens in token lists.

Scikit-learn-dependent Stages
-----------------------------

Refer to submodule `pdpipe.sklearn_stages`

* Encode - Encode a categorical column to corresponding number values.
* Scale - Scale data with any of the sklearn scalers.
* TfidfVectorizeTokenLists - Transform a column of token lists into the correponding set of tfidf vector columns.

nltk-dependent Stages
---------------------

Refer to submodule `pdpipe.nltk_stages`

* TokenizeWords - Tokenize a sentence into a list of tokens by whitespaces.
* UntokenizeWords - Joins token lists into whitespace-seperated strings.
* RemoveStopwords - Remove stopwords from a tokenized list.
* SnowballStem - Stems tokens in a list using the Snowball stemmer.
* DropRareTokens - Drop rare tokens from token lists.


Creating additional stages
==========================

Extending PdPipelineStage
-------------------------

To use other stages than the built-in ones (see [Types of Pipeline Stages](#types-of-pipeline-stages)) you can extend the  class. The constructor must pass the `PdPipelineStage` constructor the `exmsg`, `appmsg` and `desc` keyword arguments to set the exception message, application message and description for the pipeline stage, respectively. Additionally, the `_prec` and `_transform` abstract methods must be implemented to define the precondition and the effect of the new pipeline stage, respectively.

Fittable custom pipeline stages should implement, additionally to the  method, the `_fit_transform` method, which should both fit pipeline stage by the input dataframe and transform transform the dataframe, while also setting `self.is_fitted = True`.


Ad-Hoc Pipeline Stages
----------------------

To create a custom pipeline stage without creating a proper new class, you can instantiate the  class which takes a function in its `transform` constructor parameter to define the stage's operation, and the optional `prec` parameter to define a precondition (an always-true function is the default).

