# More about pdpipe

## Features


!!! python-note "Compatible with Python 3+"

    Python 3.7 and up. Crucial for new or forward-looking projects.

!!! done "Fully documented"

    Every pipeline stage and parameter are meticulously documented and accompanied by working code examples.

!!! done "Zero configuration"

    Pdpipe stages use sensible defaults for everything. Get things going immediately, tune only what you need.

!!! done "Handle mixed-type data"

    Easily create pipelines that process different types of data separately without breaking, enabling easier use of stacking-based ensemble models down the pipeline.


!!! done "Customizable stages"

    Pipeline stages are highly configurable, and creating new custom stages is easy.

!!! done "Chainable constructors & pipeline arithmetics"

    Chaining pipeline stages constructor calls for easy, one-liners creating complex pipelines. Supports pipeline arithmetics.

!!! done "Built for productization"


    Pipelines and stages are written with productization in mind; fit on training data, serialize, deserialize and transform in production.

!!! done "Fully tested"

    Pdpipe is thoroughly tested on Linux, macOS and Windows systems, as well as all Python development branches, and boasts full test coverage.

!!! done "Verbose"

    Informative prints and errors on pipeline application, including smart pre-conditions before application and post-conditions to validate successful application.


## Design Decisions

* **Extra informative naming:** Meant to make pipelines very readable, understanding their entire flow by pipeline stages names; e.g. ColDrop vs. ValDrop instead of an all-encompassing Drop stage emulating the `pandas.DataFrame.drop` method.
* **Data science-oriented naming** (rather than statistics).
* **A functional approach:** Pipelines never change input DataFrames. Nothing is done "in place".
* **Opinionated operations:** Help novices avoid mistake by default appliance of good practices; e.g., one-hot-encoding (creating dummy variables) a column will drop one of the resulting columns by default, to avoid [the dummy variable trap](http://www.algosome.com/articles/dummy-variable-trap-regression.html) (perfect [multicollinearity](https://en.wikipedia.org/wiki/Multicollinearity)).
* **Machine learning-oriented:** The target use case is transforming tabular data into a vectorized dataset on which a machine learning model will be trained; e.g., column transformations will drop the source columns to avoid strong linear dependence.

