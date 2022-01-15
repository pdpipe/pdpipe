"""shell
pip install pdpipe
"""
import pandas as pd
import pdpipe as pdp


print(pdp.__version__)

df = pd.DataFrame(
    data=[[1, 2, 'a'], [2, 4, 'b']],
    index=[1, 2],
    columns=['num1', 'num2', 'char'],
)

"""
You can easily access various pdpipe stages directly under the package handle.
"""

stage = pdp.ColDrop('num1')

"""
You don't have to build a whole pipeline to apply a stage. You can apply it as
a callable:
"""

res_df = stage(df)
print(res_df.columns)

"""
Or you can directly call the `fit_transform` method:
"""

res_df = stage.fit_transform(df)
print(res_df.columns)

"""
To get a pipeline object, provide the PdPipeline constructor with a list of
pipeline stages:
"""

pipeline = pdp.PdPipeline([stage])
print(pipeline)
res_df = pipeline.fit_transform(df, verbose=True)
print(res_df.columns)
