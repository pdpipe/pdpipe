"""Custom exceptions for pdpipe."""


class FailedPreconditionError(Exception):
    """An exception raised when a pipeline stage is applied to a dataframe for
    which the stage precondition does not hold.
    """
    pass


class PipelineApplicationError(Exception):
    """An exception raised when an exception is thrown during the application
    of a pipeline or a pipeline stage to a dataframe.
    """
    pass
