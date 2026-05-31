# pdpipe Pipelines

## Creating Pipelines

Pipelines can be created by supplying a list of pipeline stages:

<!--phmdoctest-setup-->

```python
import pdpipe as pdp
pipeline = pdp.PdPipeline([pdp.ColDrop("Name"), pdp.OneHotEncode("Label")]
```

Additionally, the `make_pdpipeline` method can be used to give stages as positional arguments.

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

## Exporting Pipeline Graphs

`PdPipeline.to_dot()` returns a dependency-free Graphviz DOT representation of
the pipeline. Node labels include each stage's index, class name, optional
stage name, and description. The returned string can be written to a `.dot`
file or passed to Graphviz tooling.

<!--phmdoctest-skip-->

```python
>>> pipeline = pdp.ColDrop("Name").OneHotEncode("Label")
>>> print(pipeline.to_dot())
digraph PdPipeline {
  graph [rankdir=LR];
  node [shape=box];
  stage_0 [label="[0] ColDrop\nDrop columns 'Name'"];
  stage_1 [label="[1] OneHotEncode\nOne-hot encode 'Label'"];
  stage_0 -> stage_1;
}
```

## Pipeline Arithmetics

Alternatively, you can create pipelines by adding pipeline stages together:

```python
pipeline = pdp.ColDrop("Name") + pdp.OneHotEncode("Label")
```

Or even by adding pipelines together or pipelines to pipeline stages:

```python
pipeline = pdp.ColDrop("Name") + pdp.OneHotEncode("Label")
pipeline += pdp.MapColVals("Job", {"Part": True, "Full": True, "No": False})
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

Assigning the parameter to a pipeline apply call with a bool sets or unsets exception raising on failed preconditions for all contained stages:

```python
pipeline = pdp.ColDrop("Name") + pdp.OneHotEncode("Label")
res_df = pipeline.apply(df, exraise=False)
```

Additionally, passing `verbose=True` to a pipeline apply call will apply all pipeline stages verbosely:

```python
res_df = pipeline.apply(df, verbose=True)
```

Finally, `fit`, `transform` and `fit_transform` all call the corresponding pipeline stage methods of all stages composing the pipeline.

## Tracing Pipeline Application

`PdPipeline.trace()` applies a deep-copied pipeline to a deep-copied dataframe
and returns structured records describing each visited stage. It is useful for
debugging pipeline behavior without fitting or mutating the original pipeline
or input dataframe.

Each trace entry includes the stage index, class, optional name, description,
status, skip reason, input/output shapes, input/output columns, and error
details if the stage failed.

<!--phmdoctest-skip-->

```python
>>> pipeline = pdp.ColDrop("Name").OneHotEncode("Label")
>>> trace = pipeline.trace(df)
>>> first_stage = trace[0]
>>> first_stage.get("status")
'applied'
>>> first_stage.get("input_columns")
['Name', 'Label', 'Children']
>>> first_stage.get("output_columns")
['Label', 'Children']
```
