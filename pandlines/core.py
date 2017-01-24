"""Defines pipelines for processing Pandas.DataFrame-based datasets."""


class PipelineStage(object):
    """Define a stage of dataframe-processing pipeline.

        Parameters
        ----------
        requirements : callable
            A callable that returns a boolean value. Used to determine whether
            this page can be applied to a given dataframe. Defaults to a
            function always returning True.
        """

    def __init__(self, requirements=lambda x: True):
        self.requirements = requirements

    def _operation(self, df):
        return df

    def apply(self, df):
        """Applies this pipeline stage to the given dataframe.

        Parameters
        ----------
        df : pandas.DataFrame
            The daraframe to which this pipeline stage will be applied.

        Returns
        -------
        pandas.DataFrame
            The resulting dataframe.
        """
        if self.requirements:
            return self._operation(df)
        return df

    __call__ = apply
