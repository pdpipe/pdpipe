# Runtime Parameters

In some cases one wishes to set a parameter based on calculations done
in real time, based on the given input. Dynamic parameters help to achieve
this with the `pdp.dynamic` function. The function is provided with a callable
which implements the logic on deciding the parameter, and is applied only when
the input is available.

## Example

The `scaling_decider` function implements the logic for choosing the parameter
type (`'StandardScaler'` or `'MinMaxScaler'`) and is passed to the
`pdp.dynamic` function in the stage's (`Scale`) constructor.
The logic references the given input and chooses a parameter based on it.

```python
import numpy as np
import pandas as pd
import pdpipe as pdp


def scaling_decider(X: pd.DataFrame) -> str:
    """
    Determines what scaler to apply by examining all numerical columns.
    """
    numX = X.select_dtypes(include=np.number)
    for col in numX.columns:
        if np.std(numX[col]) > 2 * np.mean(numX[col]):
            return "StandardScaler"
    return "MinMaxScaler"


pipeline = pdp.PdPipeline(
    stages=[
        pdp.ColDrop(pdp.cq.StartWith("n_")),  # unrelated to scaling
        pdp.Scale(
            scaler=pdp.dynamic(scaling_decider, fit=False),
            joint=True,
        ),
    ]
)
```

## Contextual Parameters

`pdp.contextual` creates a placeholder for a stage constructor parameter whose
value is produced earlier in the same pipeline application. The placeholder is
resolved from the pipeline fit context by default. Use `fit=False` to resolve
it from the application context on every pipeline application.

The stage keeps the placeholder between applications; the concrete value is
only swapped in while that stage is running. Contextual placeholders can appear
as direct constructor attributes, or inside simple built-in containers and
mapping objects stored as constructor attributes. Treat those contextual
constructor attributes as immutable parameter templates: mutations made to
them while a stage is running are discarded when the original placeholders are
restored.

```python
import numpy as np
import pandas as pd
import pdpipe as pdp


def scaling_decider(X: pd.DataFrame) -> str:
    """
    Determines what scaler to apply by examining all numerical columns.
    """
    numX = X.select_dtypes(include=np.number)
    for col in numX.columns:
        if np.std(numX[col]) > 2 * np.mean(numX[col]):
            return "StandardScaler"
    return "MinMaxScaler"


pipeline = pdp.PdPipeline(
    stages=[
        pdp.ApplicationContextEnricher(scaling_type=scaling_decider),
        pdp.Scale(
            scaler=pdp.contextual("scaling_type", fit=False),
            joint=True,
        ),
    ]
)
```
