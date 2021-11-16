"""Wrapper-kind pdpipe pipeline stages."""


from pdpipe.core import PdPipelineStage


class FitOnly(PdPipelineStage):
    """A wrapper that applies a stage to input data only when fitting.

    In other words, the input data is not transformed if the stage has
    already been fitted once.

    Parameters
    ----------
    stage : PdPipelineStage
        The pipeline stage to operate on input data only when fitting.

    Example
    -------
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
    """
    _FITONLY_DESC = "Applying, only on fit, the stage: {}"

    def __init__(self, stage, **kwargs):
        self._stage = stage
        desc = FitOnly._FITONLY_DESC.format(stage.description())
        super_kwargs = {
            'desc': desc,
        }
        super_kwargs.update(**kwargs)
        super().__init__(**super_kwargs)

    def _prec(self, df):
        if self.is_fitted:
            return True
        return self._stage._prec(df)

    def _fit_transform(self, df, verbose):
        self.is_fitted = True
        return self._stage.fit_transform(df, verbose=verbose)

    def _transform(self, df, verbose):
        if verbose:
            print(
                f"Skipping, because not in fit, "
                f"the stage: {self._stage.description()}"
            )
        return df
