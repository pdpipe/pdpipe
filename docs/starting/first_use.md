# Starting out with `pdpipe`

So how does using `pdpipe` looks like? Let's first import `pandas` and `pdpipe`, an intialize a nice little dataframe:

```python
import pandas as pd
import pdpipe as pdp

df = pd.DataFrame(
    data=[
        [23, 'Jo', 'M', True, 0.07, 'USA', 'Living life to its fullest'],
        [23, 'Dana', 'F', True, 0.3, 'USA', 'the pen is mightier then the sword'],
        [25, 'Bo', 'M', False, 2.3, 'Greece', 'all for one and one for all'],
        [44, 'Derek', 'M', True, 1.1, 'Denmark', 'every life is precious'],
        [72, 'Regina', 'F', True, 7.1, 'Greece', 'all of you get off my porch'],
        [50, 'Jim', 'M', False, 0.2, 'Germany', 'boy do I love dogs and cats'],
        [80, 'Richy', 'M', False, 100.2, 'Finland', 'I gots the dollarz'],
        [80, 'Wealthus', 'F', False, 123.2, 'Finland', 'me likey them moniez'],
    ],
    columns=['Age', 'Name', 'Gender', 'Smoking', 'Savings', 'Country', 'Quote'],
)
```

This results in the following dataframe:

|    |   Age | Name     | Gender   | Smoking   |   Savings | Country   | Quote                              |
|---:|------:|:---------|:---------|:----------|----------:|:----------|:-----------------------------------|
|  0 |    23 | Jo       | M        | True      |      0.07 | USA       | Living life to its fullest         |
|  1 |    23 | Dana     | F        | True      |      0.3  | USA       | the pen is mightier then the sword |
|  2 |    25 | Bo       | M        | False     |      2.3  | Greece    | all for one and one for all        |
|  3 |    44 | Derek    | M        | True      |      1.1  | Denmark   | every life is precious             |
|  4 |    72 | Regina   | F        | True      |      7.1  | Greece    | all of you get off my porch        |
|  5 |    50 | Jim      | M        | False     |      0.2  | Germany   | boy do I love dogs and cats        |
|  6 |    80 | Richy    | M        | False     |    100.2  | Finland   | I gots the dollarz                 |
|  7 |    80 | Wealthus | F        | False     |    123.2  | Finland   | me likey them moniez               |

## Constructing pipelines

We can create different pipeline stage object by calling their constructors,
which can be of course identified by their camel-cased names, such as 
`pdp.ColDrop` for dropping columns and `pdp.Encode` to encode them, etc.

To build a pipeline, we will usually call the `PdPipeline` class constructor,
and provide it with a list of pipeline stage objects:

```python
pipeline = pdp.PdPipeline([
    pdp.ColDrop('Name'),
    pdp.drop_rows_where['Savings'] > 100,
    pdp.Bin({'Savings': [1]}, drop=False),
    pdp.Scale('StandardScaler'),
    pdp.TokenizeText('Quote'),
    pdp.SnowballStem('EnglishStemmer', columns=['Quote']),
    pdp.RemoveStopwords('English', 'Quote'),
    pdp.Encode('Gender'),
    pdp.OneHotEncode('Country'),
])
```

??? tip "Chaining constructor syntax"

	`pdpipe` also has a chaining syntax that you can use to construct pipelines
    with characteristic one-liners (although admittedly, it is mainly
    convenient for the creation of simple, short pipelines, in dynamic Python
    shells):

	```python
	pipeline = pdp.ColDrop('Name').RowDrop({'Savings': lambda x: x > 100}) \
        .Bin({'Savings': [1]}, drop=False).Scale('StandardScaler') \
		.TokenizeText('Quote').SnowballStem('EnglishStemmer', columns=['Quote']) \
        .RemoveStopwords('English', 'Quote').Encode('Gender').OneHotEncode('Country')
	```

    **Note:** All pipeline stage constructors are available in this way, but
    some advanced handles such as `df`, and fly handles such as
    `drop_rows_where`, are not available through this syntax.


Printing the pipeline object displays it in order. 

```python
print(pipeline)
```

```bash
A pdpipe pipeline:
[ 0]  Drop columns Name
[ 1]  Drop rows in columns Savings by conditions
[ 2]  Bin Savings by [1].
[ 3]  Scale columns Columns of dtypes <class 'numpy.number'>
[ 4]  Tokenize Quote
[ 5]  Stemming tokens in Quote...
[ 6]  Remove stopwords from Quote
[ 7]  Encode Gender
[ 8]  One-hot encode Country
```

!!! tip "Pipeline slicing"

    The numbers presented in square brackets are the indices of the
    corresponding pipeline stages, and they can be used to retrieve either the
    specific pipeline stage objects composing the pipeline, e.g. with
    ```#!python pipeline[5]```, or sub-pipelines composed of sub-sequences of
    the pipeline, e.g. with ```#!python pipeline[2:6]```.


## Applying pipelines

The pipeline can now be applied to an input dataframe using the `apply`
method. We will also provide the `verbose` keyword with `True` to have a 
informative prints or the progress of dataframe processing, stage by stage:

```python
res = pipeline(df, verbose=True)
```

```bash
- Drop columns Name
- Drop rows by qualifier <RowQualifier: Qualify rows with df[Savings] >
  100>
2 rows dropped.
- Bin Savings by [1].
Savings: 100%|████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 110.24it/s]
- Scale columns Columns of dtypes <class 'numpy.number'>
- Tokenize Quote
- Stemming tokens in Quote...
- Remove stopwords from Quote
- Encode Gender
100%|█████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 204.45it/s]
- One-hot encode Country
Country: 100%|████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 140.72it/s]
```

We will thus get the dataframe below. We can see all numerical columns were scaled, the `Country` column was one-hot-encoded, `Savings` also got a binned version and the textual `Quote` column underwent some word-level manipulations:

|    |        Age |   Gender | Smoking   |   Savings | Savings_bin   | Quote                         |   Country_Germany |   Country_Greece |   Country_USA |
|---:|-----------:|---------:|:----------|----------:|:--------------|:------------------------------|------------------:|-----------------:|--------------:|
|  0 | -1.13505   |        1 | True      | -0.609615 | <1            | ['live', 'life', 'fullest']   |                 0 |                0 |             1 |
|  1 | -1.13505   |        0 | True      | -0.60482  | <1            | ['pen', 'mightier', 'sword']  |                 0 |                0 |             1 |
|  2 | -1.04979   |        1 | False     | -0.563121 | 1≤            | ['one', 'one']                |                 0 |                1 |             0 |
|  3 | -0.2398    |        1 | True      | -0.58814  | 1≤            | ['everi', 'life', 'precious'] |                 0 |                0 |             0 |
|  4 |  0.95387   |        0 | True      | -0.463043 | 1≤            | ['get', 'porch']              |                 0 |                1 |             0 |
|  5 |  0.0159866 |        1 | False     | -0.606905 | <1            | ['boy', 'love', 'dog', 'cat'] |                 1 |                0 |             0 |


## Fit and transform

Pipelines are also callable objects themselves, so calling `pipeline(df)` is 
equivalent to calling `pipeline.apply(df)`.

Additionally, pipelines inherently have a fit state. If none of the stages
composing them is fittable in nature this doesn't make a lot of a difference,
but many stage have a `fit_transform` vs `transform` logic, like encoders,
scalers and so forth.

!!! tip "`apply()` vs `fit_transform()`"

    The `apply` pipeline method uses either `fit_transform` and `transform`
    in an intelligent and sensible way: If the pipeline is not fitted, calling
    it is equivalent to calling `fit_transform`, while if it is fitted, the
    call is practically a `transform` call.

Let's say we want to utilize pdpipe's powerful slicing syntax to apply only
*some* of the pipeline stages to the raw dataframe. We will now use the 
`fit_transform` method of the pipeline itself to force all encompassed pipeline
stages to fit-transform themselves:

Here, we will use `pipeline[2:5]` to apply the binning, scaling and
tokenization stages only:

```python
pipeline[2:4].fit_transform(df)
```

|    |        Age | Name     | Gender   | Smoking   |   Savings | Savings_bin   | Country   | Quote                              |
|---:|-----------:|:---------|:---------|:----------|----------:|:--------------|:----------|:-----------------------------------|
|  0 | -1.13505   | Jo       | M        | True      | -0.609615 | <1            | USA       | Living life to its fullest         |
|  1 | -1.13505   | Dana     | F        | True      | -0.60482  | <1            | USA       | the pen is mightier then the sword |
|  2 | -1.04979   | Bo       | M        | False     | -0.563121 | 1≤            | Greece    | all for one and one for all        |
|  3 | -0.2398    | Derek    | M        | True      | -0.58814  | 1≤            | Denmark   | every life is precious             |
|  4 |  0.95387   | Regina   | F        | True      | -0.463043 | 1≤            | Greece    | all of you get off my porch        |
|  5 |  0.0159866 | Jim      | M        | False     | -0.606905 | <1            | Germany   | boy do I love dogs and cats        |
|  6 |  1.29492   | Richy    | M        | False     |  1.47805  | 1≤            | Finland   | I gots the dollarz                 |
|  7 |  1.29492   | Wealthus | F        | False     |  1.95759  | 1≤            | Finland   | me likey them moniez               |
