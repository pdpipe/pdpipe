"""Custom exceptions for pdpipe."""


class FailedPreconditionError(Exception):
    """An exception raised when a pipeline stage is applied to a dataframe for
    which the stage precondition does not hold.
    """


class FailedPostconditionError(Exception):
    """An exception raised when an expected post-condition does not hold after
    and application of the apipine stage to a dataframe.
    """


class PipelineApplicationError(Exception):
    """An exception raised when an exception is thrown during the application
    of a pipeline or a pipeline stage to a dataframe.
    """


class UnfittedPipelineStageError(Exception):
    """An exception raised when a (non-fit) transform is attempted with an
    unfitted pipeline stage.
    """
