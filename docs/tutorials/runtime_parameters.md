# Dynamic Runtime Parameters
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
import numpy as np; import pandas as pd; import pdpipe as pdp;

def scaling_decider(X: pd.DataFrame) -> str:
    """
    Determines which type of scaling to apply by examining all numerical 
    columns.
    """
    numX = X.select_dtypes(include=np.number)
    for col in numX.columns:
        if np.std(numX[col]) > 2 * np.mean(numX[col]):
            return 'StandardScaler'
    return 'MinMaxScaler'

pipeline = pdp.PdPipeline(stages=[
    pdp.ColDrop(pdp.cq.StartWith('n_')),  # not connected to scale
    pdp.Scale(
        scaler=pdp.dynamic(scaling_decider, fit=False),  
        joint=True,
    )
])
```
