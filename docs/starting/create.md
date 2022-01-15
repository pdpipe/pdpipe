# Creating additional stages

## Extending PdPipelineStage

To use other stages than the built-in ones (see [Types of Pipeline Stages](#types-of-pipeline-stages)) you can extend the  class. The constructor must pass the `PdPipelineStage` constructor the `exmsg`, `appmsg` and `desc` keyword arguments to set the exception message, application message and description for the pipeline stage, respectively. Additionally, the `_prec` and `_transform` abstract methods must be implemented to define the precondition and the effect of the new pipeline stage, respectively.

Fittable custom pipeline stages should implement, additionally to the  method, the `_fit_transform` method, which should both fit pipeline stage by the input dataframe and transform transform the dataframe, while also setting `self.is_fitted = True`.


## Ad-Hoc Pipeline Stages

To create a custom pipeline stage without creating a proper new class, you can instantiate the  class which takes a function in its `transform` constructor parameter to define the stage's operation, and the optional `prec` parameter to define a precondition (an always-true function is the default).
