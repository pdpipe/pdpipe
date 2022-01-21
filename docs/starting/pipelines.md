# pdpipe Pipelines

## Creating Pipelines

Pipelines can be created by supplying a list of pipeline stages:

<!--phmdoctest-setup-->    
```python
    import pdpipe as pdp
    pipeline = pdp.PdPipeline([pdp.ColDrop("Name"), pdp.OneHotEncode("Label")]
```

Additionally, the  method can be used to give stages as positional arguments.

```python
    pipeline = pdp.make_pdpipeline(pdp.ColDrop("Name"), pdp.OneHotEncode("Label"))
```


## Printing Pipelines

A pipeline structure can be clearly displayed by printing the object:

<!--phmdoctest-skip--> 
```python
    >>> drop_name = pdp.ColDrop("Name")
    >>> binar_label = pdp.OneHotEncode("Label")
    >>> map_job = pdp.MapColVals("Job", {"Part": True, "Full":True, "No": False})
    >>> pipeline = pdp.PdPipeline([drop_name, binar_label, map_job])
    >>> print(pipeline)
    A pdpipe pipeline:
    [ 0]  Drop column Name
    [ 1]  OneHotEncode Label
    [ 2]  Map values of column Job with {'Part': True, 'Full': True, 'No': False}
```

## Pipeline Arithmetics

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

## Pipeline Chaining

Pipeline stages can also be chained to other stages to create pipelines:

```python
  pipeline = pdp.ColDrop("Name").OneHotEncode("Label").ValDrop([-1], "Children")
```

## Pipeline Slicing

Pipelines are Python Sequence objects, and as such can be sliced using Python's slicing notation, just like lists:

<!--phmdoctest-skip--> 
```python
  >>> pipeline = pdp.ColDrop("Name").OneHotEncode("Label").ValDrop([-1], "Children").ApplyByCols("height", math.ceil)
  >>> pipeline[0]
  Drop column Name
  >>> pipeline[1:2]
  A pdpipe pipeline:
  [ 0] OneHotEncode Label
```

Pipelines can also be sliced by the stages `name` parameter, notice when running `pipeline[['name1', 'name2']]` a new pipeline will returned with all stages that they `name` is 'name1' or 'name2', and when running `pipeline['name']` only the first stage that has the 'name' will return.:

```python
  >>> pipeline = pdp.ColDrop("Name", name="dropName").OneHotEncode("Label", name="encoder").ValDrop([-1], "Children").ApplyByCols("height", math.ceil)
  >>> pipeline['dropName']
  PdPipelineStage: Drop columns Name
  >>> pipeline[['dropName', 'encoder']]
  A pdpipe pipeline:
  [ 0]  Drop columns Name
  [ 1]  One-hot encode Label
```

## Applying Pipelines

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
