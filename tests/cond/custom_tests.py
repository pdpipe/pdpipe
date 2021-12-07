"""Custom tests of pdpipe conditions meant to run manually."""

import pandas as pd
import pdpipe as pdp


if __name__ == "__main__":
    df = pd.DataFrame([[1, 4], [4, 5], [1, 11]], [1, 2, 3], ['a', 'b'])
    pline = pdp.PdPipeline([
        pdp.FreqDrop(2, 'a', prec=pdp.cond.HasAllColumns(['x']))
    ])
    pline.apply(df, verbose=True)
