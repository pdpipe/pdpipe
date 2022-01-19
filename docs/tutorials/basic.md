# Basic Usage

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

Now, let's build a pipeline with an impressive one-liner, utilizing the chaining constructor capabilities of `pdpipe`:

```python
pipeline = pdp.ColDrop('Name').RowDrop({'Savings': lambda x: x > 100}).Bin({'Savings': [1]}, drop=False).Scale(
    'StandardScaler').TokenizeText('Quote').SnowballStem('EnglishStemmer', columns=['Quote']).RemoveStopwords(
	    'English', 'Quote').Encode('Gender').OneHotEncode('Country')
```

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
