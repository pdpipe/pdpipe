
## Ad-Hoc Pipeline Stages

To create a custom pipeline stage without creating a proper new class, you can instantiate the  class which takes a function in its `transform` constructor parameter to define the stage's operation, and the optional `prec` parameter to define a precondition (an always-true function is the default).

!!! code-example "Creating an AdHoc pdpipe stage"

    ```python
    test_stage = AdHocStage(
		transform=lambda df: df.drop(['num'], axis=1),
		prec=lambda df: 'num' in df.columns
	)
    ```

!!! warning 

    Since Python lambdas are not serializable by Python `pickle`, note that
    using a lambda as the transform operator of an AdHocStage will make
    unpickleable.
