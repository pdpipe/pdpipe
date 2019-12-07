pdpipe ˨ 
########

|PyPI-Status| |Downloads| |PyPI-Versions| |Build-Status| |Codecov| |Codefactor| |LICENCE|


Easy pipelines for pandas DataFrames (`learn how! <https://towardsdatascience.com/https-medium-com-tirthajyoti-build-pipelines-with-pandas-using-pdpipe-cade6128cd31>`_).

.. code-block:: python

  >>> df = pd.DataFrame(
          data=[[4, 165, 'USA'], [2, 180, 'UK'], [2, 170, 'Greece']],
          index=['Dana', 'Jane', 'Nick'],
          columns=['Medals', 'Height', 'Born']
      )
  >>> import pdpipe as pdp
  >>> pipeline = pdp.ColDrop('Medals').OneHotEncode('Born')
  >>> pipeline(df)
              Height  Born_UK  Born_USA
      Dana     165        0         1
      Jane     180        1         0
      Nick     170        0         0

.. .. alternative symbols: ˨ ᛪ ᛢ ᚶ ᚺ ↬ ⑀ ⤃ ⤳ ⥤ 』

.. contents::

.. section-numbering::

Installation
============

Install ``pdpipe`` with:

.. code-block:: bash

  pip install pdpipe

Some pipeline stages require ``scikit-learn``; they will simply not be loaded if ``scikit-learn`` is not found on the system, and ``pdpipe`` will issue a warning. To use them you must also `install scikit-learn <http://scikit-learn.org/stable/install.html>`_.


Similarly, some pipeline stages require ``nltk``; they will simply not be loaded if ``nltk`` is not found on your system, and ``pdpipe`` will issue a warning. To use them you must additionally `install nltk <http://www.nltk.org/install.html>`_.


Features
========

* A simple interface.
* Informative prints and errors on pipeline application.
* Chaining pipeline stages constructor calls for easy, one-liners pipelines.
* Pipeline arithmetics.
* Easier handling of mixed data (numeric, categorical and others).
* `Fully tested on Linux, macOS and Windows systems <https://travis-ci.org/shaypal5/pdpipe>`_.
* Compatible with Python 3.5+.
* Pure Python.


Design Decisions
----------------

* **Extra infromative naming:** Meant to make pipelines very readable, understanding their entire flow by pipeline stages names; e.g. ColDrop vs. ValDrop instead of an all-encompassing Drop stage emulating the ``pandas.DataFrame.drop`` method.
* **Data science-oriented naming** (rather than statistics).
* **A functional approach:** Pipelines never change input DataFrames. Nothing is done "in place".
* **Opinionated operations:** Help novices avoid mistake by default appliance of good practices; e.g., one-hot-encoding (creating dummy variables) a column will drop one of the resulting columns by default, to avoid `the dummy variable trap`_ (perfect `multicollinearity`_).
* **Machine learning-oriented:** The target use case is transforming tabular data into a vectorized dataset on which a machine learning model will be trained; e.g., column transformations will drop the source columns to avoid strong linear dependence.

.. _`the dummy variable trap`: http://www.algosome.com/articles/dummy-variable-trap-regression.html
.. _`multicollinearity`: https://en.wikipedia.org/wiki/Multicollinearity


Basic Use
=========

The awesome Tirthajyoti Sarkar wrote `an excellent practical introduction on how to use pdpipe <https://towardsdatascience.com/https-medium-com-tirthajyoti-build-pipelines-with-pandas-using-pdpipe-cade6128cd31>`_. Read it now `on Towards Data Science <https://towardsdatascience.com/https-medium-com-tirthajyoti-build-pipelines-with-pandas-using-pdpipe-cade6128cd31>`_!

Pipeline Stages
---------------

Creating Pipeline Stages
~~~~~~~~~~~~~~~~~~~~~~~~

You can create stages with the following syntax:

.. code-block:: python

  import pdpipe as pdp
  drop_name = pdp.ColDrop("Name")


All pipeline stages have a predefined precondition function that returns True for dataframes to which the stage can be applied. By default, pipeline stages raise an exception if a DataFrame not meeting their precondition is piped through. This behaviour can be set per-stage by assigning ``exraise`` with a bool in the constructor call. If ``exraise`` is set to ``False`` the input DataFrame is instead returned without change:

.. code-block:: python

  drop_name = pdp.ColDrop("Name", exraise=False)


Applying Pipeline Stages
~~~~~~~~~~~~~~~~~~~~~~~~

You can apply a pipeline stage to a DataFrame using its ``apply`` method:

.. code-block:: python

  res_df = pdp.ColDrop("Name").apply(df)

Pipeline stages are also callables, making the following syntax equivalent:

.. code-block:: python

  drop_name = pdp.ColDrop("Name")
  res_df = drop_name(df)

The initialized exception behaviour of a pipeline stage can be overridden on a per-application basis:

.. code-block:: python

  drop_name = pdp.ColDrop("Name", exraise=False)
  res_df = drop_name(df, exraise=True)

Additionally, to have an explanation message print after the precondition is checked but before the application of the pipeline stage, pass ``verbose=True``:

.. code-block:: python

  res_df = drop_name(df, verbose=True)

All pipeline stages also adhere to the ``scikit-learn`` transformer API, and so have ``fit_transform`` and ``transform`` methods; these behave exactly like ``apply``, and accept the input dataframe as parameter ``X``. For the same reason, pipeline stages also have a ``fit`` method, which applies them but returns the input dataframe unchanged.


Fittable Pipeline Stages
~~~~~~~~~~~~~~~~~~~~~~~~

Some pipeline stages can be fitted, meaning that some transformation parameters are set the first time a dataframe is piped through the stage, while later applications of the stage use these now-set parameters without changing them; the ``Encode`` scikit-learn-dependent stage is a good example.

For these type of stages the first call to ``apply`` will both fit the stage and transform the input dataframe, while subsequent calls to ``apply`` will transform input dataframes according to the already-fitted transformation parameters.

Additionally, for fittable stages the ``scikit-learn`` transformer API methods behave as expected:

* ``fit`` sets the transformation parameters of the stage but returns the input dataframe unchanged.
* ``fit_transform`` both sets the transformation parameters of the stage and returns the input dataframe after transformation.
* ``transform`` transforms input dataframes according to already-fitted transformation parameters; if the stage is not fitted, an ``UnfittedPipelineStageError`` is raised.

Again, ``apply``, ``fit_transform`` and ``transform`` are all of equivalent for non-fittable pipeline stages. And in all cases the ``y`` parameter of these methods is ignored.


Pipelines
---------

Creating Pipelines
~~~~~~~~~~~~~~~~~~

Pipelines can be created by supplying a list of pipeline stages:

.. code-block:: python

  pipeline = pdp.PdPipeline([pdp.ColDrop("Name"), pdp.OneHotEncode("Label")])

Additionally, the ``make_pdpipeline`` method can be used to give stages as positional arguments.

.. code-block:: python

    pipeline = pdp.make_pdpipeline(pdp.ColDrop("Name"), pdp.OneHotEncode("Label"))


Printing Pipelines
~~~~~~~~~~~~~~~~~~

A pipeline structre can be clearly displayed by printing the object:

.. code-block:: python

  >>> drop_name = pdp.ColDrop("Name")
  >>> binar_label = pdp.OneHotEncode("Label")
  >>> map_job = pdp.MapColVals("Job", {"Part": True, "Full":True, "No": False})
  >>> pipeline = pdp.PdPipeline([drop_name, binar_label, map_job])
  >>> print(pipeline)
  A pdpipe pipeline:
  [ 0]  Drop column Name
  [ 1]  OneHotEncode Label
  [ 2]  Map values of column Job with {'Part': True, 'Full': True, 'No': False}.


Pipeline Arithmetics
~~~~~~~~~~~~~~~~~~~~

Alternatively, you can create pipelines by adding pipeline stages together:

.. code-block:: python

  pipeline = pdp.ColDrop("Name") + pdp.OneHotEncode("Label")

Or even by adding pipelines together or pipelines to pipeline stages:

.. code-block:: python

  pipeline = pdp.ColDrop("Name") + pdp.OneHotEncode("Label")
  pipeline += pdp.MapColVals("Job", {"Part": True, "Full":True, "No": False})
  pipeline += pdp.PdPipeline([pdp.ColRename({"Job": "Employed"})])


Pipeline Chaining
~~~~~~~~~~~~~~~~~

Pipeline stages can also be chained to other stages to create pipelines:

.. code-block:: python

  pipeline = pdp.ColDrop("Name").OneHotEncode("Label").ValDrop([-1], "Children")


Pipeline Slicing
~~~~~~~~~~~~~~~~

Pipelines are Python Sequence objects, and as such can be sliced using Python's slicing notation, just like lists:

.. code-block:: python

  >>> pipeline = pdp.ColDrop("Name").OneHotEncode("Label").ValDrop([-1], "Children").ApplyByCols("height", math.ceil)
  >>> pipeline[0]
  Drop column Name
  >>> pipeline[1:2]
  A pdpipe pipeline:
  [ 0] OneHotEncode Label


Applying Pipelines
~~~~~~~~~~~~~~~~~~

Pipelines are pipeline stages themselves, and can be applied to a DataFrame using the same syntax, applying each of the stages making them up, in order:

.. code-block:: python

  pipeline = pdp.ColDrop("Name") + pdp.OneHotEncode("Label")
  res_df = pipeline(df)


Assigning the ``exraise`` parameter to a pipeline apply call with a bool sets or unsets exception raising on failed preconditions for all contained stages:

.. code-block:: python

  pipeline = pdp.ColDrop("Name") + pdp.OneHotEncode("Label")
  res_df = pipeline.apply(df, exraise=False)


Additionally, passing ``verbose=True`` to a pipeline apply call will apply all pipeline stages verbosely:

.. code-block:: python

  res_df = pipeline.apply(df, verbose=True)


Finally, ``fit``, ``transform`` and ``fit_transform`` all call the corresponding pipeline stage methods of all stages composing the pipeline


Types of Pipeline Stages
========================

All built-in stages are thoroughly documented, including examples; if you find any documentation lacking please open an issue. A list of briefly described available built-in stages follows:

Basic Stages
------------

* AdHocStage - Define custom pipeline stages on the fly.
* ColDrop - Drop columns by name.
* ValDrop - Drop rows by by their value in specific or all columns.
* ValKeep - Keep rows by by their value in specific or all columns.
* ColRename - Rename columns.
* DropNa - Drop null values. Supports all parameter supported by pandas.dropna function. 
* FreqDrop - Drop rows by value frequency threshold on a specific column 
* ColReorder - Reorder columns.
* RowDrop - Drop rows by callable conditions.

Column Generation
-----------------

* Bin - Convert a continuous valued column to categoric data using binning.
* OneHotEncode - Convert a categorical column to the several binary columns corresponding to it.
* MapColVals - Replace column values by a map.
* ApplyToRows - Generate columns by applying a function to each row.
* ApplyByCols - Generate columns by applying an element-wise function to columns.
* ColByFrameFunc - Add a column by applying a dataframe-wide function.
* AggByCols - Generate columns by applying an series-wise function to columns.
* Log - Log-transform numeric data, possibly shifting data before.

Scikit-learn-dependent Stages
-----------------------------

* Encode - Encode a categorical column to corresponding number values.
* Scale - Scale data with any of the sklearn scalers. 
  

nltk-dependent Stages
---------------------

* TokenizeWords - Tokenize a sentence into a list of tokens by whitespaces.
* UntokenizeWords - Joins token lists into whitespace-seperated strings.
* RemoveStopwords - Remove stopwords from a tokenized list.
* SnowballStem - Stems tokens in a list using the Snowball stemmer.
* DropRareTokens - Drop rare tokens from token lists.


Creating additional stages
==========================

Extending PdPipelineStage
-------------------------

To use other stages than the built-in ones (see `Types of Pipeline Stages`_) you can extend the ``PdPipelineStage`` class. The constructor must pass the ``PdPipelineStage`` constructor the ``exmsg``, ``appmsg`` and ``desc`` keyword arguments to set the exception message, application message and description for the pipeline stage, respectively. Additionally, the ``_prec`` and ``_transform`` abstract methods must be implemented to define the precondition and the effect of the new pipeline stage, respectively.

Fittable custom pipeline stages should implement, additionally to the ``_transform`` method, the ``_fit_transform`` method, which should both fit pipeline stage by the input dataframe and transform transform the dataframe, while also setting ``self.is_fitted = True``. 


Ad-Hoc Pipeline Stages
----------------------

To create a custom pipeline stage without creating a proper new class, you can instantiate the ``AdHocStage`` class which takes a function in its ``transform`` constructor parameter to define the stage's operation, and the optional ``prec`` parameter to define a precondition (an always-true function is the default).


Contributing
============

Package author and current maintainer is Shay Palachy (shay.palachy@gmail.com); You are more than welcome to approach him for help. Contributions are very welcomed, especially since this package is very much in its infancy and many other pipeline stages can be added. Intuit are nice.

Installing for development
--------------------------

Clone:

.. code-block:: bash

  git clone git@github.com:shaypal5/pdpipe.git


Install in development mode with test dependencies:

.. code-block:: bash

  cd pdpipe
  pip install -e ".[test]"


Running the tests
-----------------

To run the tests, use:

.. code-block:: bash

  python -m pytest --cov=pdpipe


Adding documentation
--------------------

This project is documented using the `numpy docstring conventions`_, which were chosen as they are perhaps the most widely-spread conventions that are both supported by common tools such as Sphinx and result in human-readable docstrings (in my personal opinion, of course). When documenting code you add to this project, please follow `these conventions`_.

.. _`numpy docstring conventions`: https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard
.. _`these conventions`: https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard

Additionally, if you update this ``README.rst`` file,  use ``python setup.py checkdocs`` to validate it compiles.


Credits
=======
Created by Shay Palachy  (shay.palachy@gmail.com).

.. alternative:
.. https://badge.fury.io/py/yellowbrick.svg

.. |PyPI-Status| image:: https://img.shields.io/pypi/v/pdpipe.svg
  :target: https://pypi.org/project/pdpipe

.. |PyPI-Versions| image:: https://img.shields.io/pypi/pyversions/pdpipe.svg
   :target: https://pypi.org/project/pdpipe

.. |Build-Status| image:: https://travis-ci.org/shaypal5/pdpipe.svg?branch=master
  :target: https://travis-ci.org/shaypal5/pdpipe

.. |LICENCE| image:: https://img.shields.io/badge/License-MIT-yellow.svg
  :target: https://pypi.python.org/pypi/pdpipe
  
.. .. |LICENCE| image:: https://github.com/shaypal5/pdpipe/blob/master/mit_license_badge.svg
  :target: https://pypi.python.org/pypi/pdpipe
  
.. https://img.shields.io/pypi/l/pdpipe.svg

.. |Codecov| image:: https://codecov.io/github/shaypal5/pdpipe/coverage.svg?branch=master
   :target: https://codecov.io/github/shaypal5/pdpipe?branch=master

  
.. |Codacy|  image:: https://api.codacy.com/project/badge/Grade/7d605e063f114ecdb5569266bd0226cd
   :alt: Codacy Badge
   :target: https://app.codacy.com/app/shaypal5/pdpipe?utm_source=github.com&utm_medium=referral&utm_content=shaypal5/pdpipe&utm_campaign=Badge_Grade_Dashboard

.. |Requirements| image:: https://requires.io/github/shaypal5/pdpipe/requirements.svg?branch=master
     :target: https://requires.io/github/shaypal5/pdpipe/requirements/?branch=master
     :alt: Requirements Status

.. |Downloads| image:: https://pepy.tech/badge/pdpipe
     :target: https://pepy.tech/project/pdpipe
     :alt: PePy stats
     
.. |Codefactor| image:: https://www.codefactor.io/repository/github/shaypal5/pdpipe/badge?style=plastic
     :target: https://www.codefactor.io/repository/github/shaypal5/pdpipe
     :alt: Codefactor code quality
