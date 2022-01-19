# Stage Wrappers

`pdpipe` implementes pipeline stage wrappers that can change how pipeline stages behave.

The only example at the moment is `FitOnly`, a wrapper that applies a stage to input data only when the stage is being fitted.

!!! code-example "The FitOnly wrapper"

    ```python
    >>> import pandas as pd; import pdpipe as pdp;
    >>> df = pd.DataFrame([[8,'a'],[5,'b']], [1,2], ['num', 'char'])
    >>> stage = pdp.FitOnly(pdp.ColDrop('num'))
    >>> stage(df)
      char
    1    a
    2    b
    >>> df2 = pd.DataFrame([[8,'a'],[5,'b']], [1,2], ['num', 'char'])
    >>> stage(df2)
       num char
    1    8    a
    2    5    b
    ```
