pdpipe
#########
|PyPI-Status| |PyPI-Versions| |Build-Status| |Codecov| |LICENCE|

Easy pipelines for pandas DataFrames.

.. code-block:: python

  >>> df = pd.DataFrame(
          data=[[4, 165, 'USA'], [2, 180, 'UK'], [2, 170, 'Greece']],
          index=['Dana', 'Jack', 'Nick'],
          columns=['Medals', 'Height', 'Born']
      )
  >>> pipeline = pdp.ColDrop('Medals').Binarize('Born')
  >>> pipeline(df)
              Height  Born_UK  Born_USA
      Dana     165        0         1
      Jack     180        1         0
      Nick     170        0         0

.. contents::

.. section-numbering::

Installation
============

Install ``pdpipe`` with:

.. code-block:: bash

  pip install pdpipe

Some pipeline stages require ``scikit-learn``; they will simply not be loaded if ``scikit-learn`` is not found on the system, and ``pdpipe`` will issue a warning. To use them you must also `install scikit-learn`_.

.. _`install scikit-learn`: http://scikit-learn.org/stable/install.html


Features
========

* Pure Python.
* Compatible with Python 3.5+.
* A simple interface.
* Informative prints and errors on pipeline application.
* Chaining pipeline stages constructor calls for easy, one-liners pipelines.
* Pipeline arithmetics.


Design Decisions
----------------

* **Data science-oriented naming** (rather than statistics).
* **A functional approach:** Pipelines never change input DataFrames. Nothing is done "in place".
* **Opinionated operations:** Help novices avoid mistake by default appliance of good practices; e.g., binarizing (creating dummy variables) a column will drop one of the resulting columns by default, to avoid `the dummy variable trap`_ (perfect `multicollinearity`_).
* **Machine learning-oriented:** The target use case is transforming tabular data into a vectorized dataset on which a machine learning model will be trained; e.g., column transformations will drop the source columns to avoid strong linear dependence.

.. _`the dummy variable trap`: http://www.algosome.com/articles/dummy-variable-trap-regression.html
.. _`multicollinearity`: https://en.wikipedia.org/wiki/Multicollinearity


Use
===

Pipeline Stages
---------------

Creating Pipeline Stages
~~~~~~~~~~~~~~~~~~~~~~~~

You can create stages with the following syntax:

.. code-block:: python

  import pdpipe as pdp
  drop_name = pdp.ColDrop("Name")


All pipeline stages have a predefined precondition function that returns True for dataframes to which the stage can be applied. By default, pipeline stages raise an exception if a DataFrame not meeting
their precondition is piped through. This behaviour can be set per-stage by assigning ``exraise`` with a bool in the constructor call. If ``exraise`` is set to ``False`` the input DataFrame is instead returned without change:

.. code-block:: python

  drop_name = pdp.ColDrop("Name", exraise=False)


Applying Pipelines Stages
~~~~~~~~~~~~~~~~~~~~~~~~~

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


Fittable Pipeline Stages
~~~~~~~~~~~~~~~~~~~~~~~~

Some pipeline stages can be fitted, meaning that some transformation parameters are set the first time a dataframe is piped through the stage, while later applications of the stage use these now-set parameters without changing them; the ``Encode`` stage is a good example.

If you want to re-fit an already fitted pipeline stage use the ``fit_transform`` method to re-fit the stage to a new dataframe. Notice that for an unfitted stage ``apply`` and ``fit_transform`` are equivalent, and only later calls to apply will ``transform`` input dataframes without refitting the stage.

Finally, ``apply`` and ``fit_transform`` are of course equivalent for non-fittable pipeline stages.


Extending PipelineStage
~~~~~~~~~~~~~~~~~~~~~~~

To use other stages than the built-in ones (see `Types of Pipeline Stages`_) you can extend the ``PipelineStage`` class. The constructor must pass the ``PipelineStage`` constructor the ``exmsg``, ``appmsg`` and ``desc`` keyword arguments to set the exception message, application message and description for the pipeline stage, respectively. Additionally, the ``_prec`` and ``_op`` abstract methods must be implemented to define the precondition and the effect of the new pipeline stage, respectively.

Fittable custom pipeline stages should implement, additionally to the ``_op`` method, the ``_transform`` method, which should apply the fitted pipeline to an input dataframe, while also setting ``self.is_fitted = True``. The ``_op`` method then acts as the ``fit_tranform`` for the stage.


Ad-Hoc Pipeline Stages
~~~~~~~~~~~~~~~~~~~~~~

To create a custom pipeline stage without creating a proper new class, you can instantiate the ``AdHocStage`` class which takes a function in its ``op`` constructor parameter to define the stage's operation, and the optional ``prec`` parameter to define a precondition (an always-true function is the default).


Pipelines
---------

Creating Pipelines
~~~~~~~~~~~~~~~~~~

Pipelines can be created by supplying a list of pipeline stages:

.. code-block:: python

  pipeline = pdp.Pipeline([pdp.ColDrop("Name"), pdp.Binarize("Label")])


Pipeline Arithmetics
~~~~~~~~~~~~~~~~~~~~

Alternatively, you can create pipelines by adding pipeline stages together:

.. code-block:: python

  pipeline = pdp.ColDrop("Name") + pdp.Binarize("Label")

Or even by adding pipelines together or pipelines to pipeline stages:

.. code-block:: python

  pipeline = pdp.ColDrop("Name") + pdp.Binarize("Label")
  pipeline += pdp.ApplyToRows("Job", {"Part": True, "Full":True, "No": False})
  pipeline += pdp.Pipeline([pdp.ColRename({"Job": "Employed"})])


Pipeline Chaining
~~~~~~~~~~~~~~~~~

Pipeline stages can also be chained to other stages to create pipelines:

.. code-block:: python

  pipeline = pdp.ColDrop("Name").Binarize("Label").ValDrop([-1], "Children")


Pipeline Slicing
~~~~~~~~~~~~~~~~

Pipelines are Python Sequence objects, and as such can be sliced using Python's slicing notation, just like lists:

.. code-block:: python

  pipeline = pdp.ColDrop("Name").Binarize("Label").ValDrop([-1], "Children").ApplyByCols("height", math.ceil)
  result_df = pipeline[1:2](df)


Applying Pipelines
~~~~~~~~~~~~~~~~~~

Pipelines are pipeline stages themselves, and can be applied to a DataFrame using the same syntax, applying each of the stages making them up, in order:

.. code-block:: python

  pipeline = pdp.ColDrop("Name") + pdp.Binarize("Label")
  res_df = pipeline(df)


Assigning the ``exraise`` parameter to a pipeline apply call with a bool sets or unsets exception raising on failed preconditions for all contained stages:

.. code-block:: python

  pipeline = pdp.ColDrop("Name") + pdp.Binarize("Label")
  res_df = pipeline.apply(df, exraise=False)


Additionally, passing ``verbose=True`` to a pipeline apply call will apply all pipeline stages verbosely:

.. code-block:: python

  res_df = pipeline.apply(df, verbose=True)


Finally, to re-fit all fittable pipeline stages in the pipeline use the ``fit_transform`` method, which calls the corresponding method for all composing pipeline stages.


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
* FreqDrop - Drop rows by value frequency threshold on a specific column. 

Column Generation
-----------------

* Bin - Convert a continuous valued column to categoric data using binning.
* Binarize - Convert a categorical column to the several binary columns corresponding to it.
* ApplyToRows - Generate columns by applying a function to each row.
* ApplyByCols - Generate columns by applying an element-wise function to columns.

Scikit-learn-dependent Stages
-----------------------------

* Encode - Encode a categorical column to corresponding number values.


Contributing
============

Package author and current maintainer is Shay Palachy (shay.palachy@gmail.com); You are more than welcome to approach him for help. Contributions are very welcomed, especially since this package is very much in its infancy and many other pipeline stages can be added.

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

.. _`numpy docstring conventions`: https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt
.. _`these conventions`: https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt


Credits
=======
Created by Shay Palachy  (shay.palachy@gmail.com).


.. |PyPI-Status| image:: https://img.shields.io/pypi/v/pdpipe.svg
  :target: https://pypi.python.org/pypi/pdpipe

.. |PyPI-Versions| image:: https://img.shields.io/pypi/pyversions/pdpipe.svg
   :target: https://pypi.python.org/pypi/pdpipe

.. |Build-Status| image:: https://travis-ci.org/shaypal5/pdpipe.svg?branch=master
  :target: https://travis-ci.org/shaypal5/pdpipe

.. |LICENCE| image:: https://img.shields.io/pypi/l/pdpipe.svg
  :target: https://pypi.python.org/pypi/pdpipe

.. |Codecov| image:: https://codecov.io/github/shaypal5/pdpipe/coverage.svg?branch=master
   :target: https://codecov.io/github/shaypal5/pdpipe?branch=master
