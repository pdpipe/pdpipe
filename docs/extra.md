

## Features

* A simple interface.
* Informative prints and errors on pipeline application.
* Chaining pipeline stages constructor calls for easy, one-liners pipelines.
* Pipeline arithmetics.
* Easier handling of mixed data (numeric, categorical and others).
* [Fully tested on Linux, macOS and Windows systems](https://travis-ci.org/pdpipe/pdpipe).
* Compatible with Python 3.7+.
* Pure Python.


### Design Decisions

* **Extra informative naming:** Meant to make pipelines very readable, understanding their entire flow by pipeline stages names; e.g. ColDrop vs. ValDrop instead of an all-encompassing Drop stage emulating the `pandas.DataFrame.drop` method.
* **Data science-oriented naming** (rather than statistics).
* **A functional approach:** Pipelines never change input DataFrames. Nothing is done "in place".
* **Opinionated operations:** Help novices avoid mistake by default appliance of good practices; e.g., one-hot-encoding (creating dummy variables) a column will drop one of the resulting columns by default, to avoid [the dummy variable trap](http://www.algosome.com/articles/dummy-variable-trap-regression.html) (perfect [multicollinearity](https://en.wikipedia.org/wiki/Multicollinearity)).
* **Machine learning-oriented:** The target use case is transforming tabular data into a vectorized dataset on which a machine learning model will be trained; e.g., column transformations will drop the source columns to avoid strong linear dependence.

