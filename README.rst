pdpipe
#########
|PyPI-Status| |PyPI-Versions| |Build-Status| |Codecov| |LICENCE|

Easy pipelines for pandas.

.. code-block:: python

  >>> df = pd.DataFrame(
          data=[[4, 165, 'USA'], [2, 180, 'UK'], [2, 170, 'Greece']],
          index=['Dana', 'Jack', 'Nick'],
          columns=['Medals', 'Height', 'Born']
      )
  >>> pipeline = pdp.Coldrop('Medals').Binarize('Born')
  >>> pipline(df)
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

Some stages require ``scikit-learn``; they will simply not be loaded if ``scikit-learn`` is not found on the system, and ``pdpipe`` will issue a warning.

Use
===

Creating Pipline Stages
-----------------------

Create stages with the following syntax:

.. code-block:: python

  import pdpipde as pdp
  drop_name = pdp.ColDrop("Name")

By default, pipeline stages raise an exception if a DataFrame not meeting
their precondition is piped through. This behaviour can be set per-stage by
assigning ``exraise`` with a bool in a constructor call:

.. code-block:: python

  drop_name = pdp.ColDrop("Name", exraise=False)

Creating Piplines
-----------------

Pipelines can be created by supplying a list of pipeline stages:

.. code-block:: python

  pipeline = pdp.Pipeline([pdp.ColDrop("Name"), pdp.Binarize("Label")])

Alternatively, you can add pipeline stages together:

.. code-block:: python

  pipeline = pdp.ColDrop("Name") + pdp.Binarize("Label")

Or even by adding pipelines together or pipelines to pipeline stages:

.. code-block:: python

  pipeline = pdp.ColDrop("Name") + pdp.Binarize("Label")
  pipeline += pdp.MapColVals("Job", {"Part": True, "Full":True, "No": False})
  pipeline += pdp.Pipeline([pdp.ColRename({"Job": "Employed"})])

Pipline stages can also be chained to other stages to create pipelines:

.. code-block:: python

  pipeline = pdp.ColDrop("Name").Binarize("Label").ValDrop([-1], "Children")

Applying Pipelines Stages
-------------------------

You can apply a pipeline stage to a DataFrame using its ``apply`` method:

.. code-block:: python

  res_df = pdp.ColDrop("Name").apply(df)

Pipeline stages are also callables, making the following syntax equivalent:

.. code-block:: python

  drop_name = pdp.ColDrop("Name")
  res_df = drop_name(df)

The initialized exception behaviour of a pipeline stage can be overriden on a per-application basis:

.. code-block:: python

  drop_name = pdp.ColDrop("Name", exraise=False)
  res_df = drop_name(df, exraise=True)


Applying Pipelines
------------------

Pipelines are pipeline stages themselves, and can be applied to DataFrame using the same syntax, applying each of the stages making them up, in order:

.. code-block:: python

  pipeline = pdp.ColDrop("Name") + pdp.Binarize("Label")
  res_df = pipeline(df)


Assigning the ``exraise`` paramter to a pipeline apply call with a bool set or unsets exception raising on failed preconditions for all contained stages:

.. code-block:: python

  pipeline = pdp.ColDrop("Name") + pdp.Binarize("Label")
  res_df = pipeline.apply(df, exraise=True)


Pipeline Stages
===============

Basic Stages
------------

* ColDrop - Drop columns by name.
* ValDrop - Drop rows by by their value in specific or all columns.
* ValKeep - Keep rows by by their value in specific or all columns.
* ColRename - Rename columns.
* Bin - Convert a continous valued column to categoric data using binning.
* Binarize - Convert a categorical column to the several binary columns corresponding to it.
* MapColVals - Convert column values using a mapping.

Scikit-learn-dependent Stages
-----------------------------

* Encode - Encode a categorical column to corresponding number values.


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
