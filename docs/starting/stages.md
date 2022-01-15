# Pipeline Stages

## Creating Pipeline Stages

You can create stages with the following syntax:

```python
  import pdpipe as pdp
  drop_name = pdp.ColDrop("Name")
```

All pipeline stages have a predefined precondition function that returns True for dataframes to which the stage can be applied. By default, pipeline stages raise an exception if a DataFrame not meeting their precondition is piped through. This behaviour can be set per-stage by assigning `exraise` with a bool in the constructor call. If `exraise` is set to `False` the input DataFrame is instead returned without change:

```python
  drop_name = pdp.ColDrop("Name", exraise=False)
```


## Applying Pipeline Stages

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


## Fittable Pipeline Stages

Some pipeline stages can be fitted, meaning that some transformation parameters are set the first time a dataframe is piped through the stage, while later applications of the stage use these now-set parameters without changing them; the `Encode` scikit-learn-dependent stage is a good example.

For these type of stages the first call to `apply` will both fit the stage and transform the input dataframe, while subsequent calls to `apply` will transform input dataframes according to the already-fitted transformation parameters.

Additionally, for fittable stages the `scikit-learn` transformer API methods behave as expected:

* `fit` sets the transformation parameters of the stage but returns the input dataframe unchanged.
* `fit_transform` both sets the transformation parameters of the stage and returns the input dataframe after transformation.
* `transform` transforms input dataframes according to already-fitted transformation parameters; if the stage is not fitted, an `UnfittedPipelineStageError` is raised.

Again, `apply`, `fit_transform` and  are all of equivalent for non-fittable pipeline stages. And in all cases the `y` parameter of these methods is ignored.
