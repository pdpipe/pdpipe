"""Custom exceptions for pdpipe."""


class FailedPreconditionError(Exception):
    """An exception raised when a pipeline stage is applied to a dataframe for
    which the stage precondition does not hold.
    """


class FailedPostconditionError(Exception):
    """An exception raised when an expected post-condition does not hold after
    an application of the a pipeline stage to a dataframe.
    """


class FailedConditionError(Exception):
    """An exception raised when an expected condition does not hold for an
    input dataframe.
    """


class PipelineInitializationError(Exception):
    """An exception raised when a pipeline is not initialized properly."""


class PipelineApplicationError(Exception):
    """An exception raised when an exception is thrown during the application
    of a pipeline or a pipeline stage to a dataframe.
    """


class UnfittedPipelineStageError(Exception):
    """An exception raised when a (non-fit) transform is attempted with an
    unfitted pipeline stage.
    """


class UnexpectedPipelineMethodCallError(Exception):
    """An exception raised when a placeholder method implementation of an
    extension of the PdPipeline class is called when it was not expected.

    If this exception is raised, it is very likely a result of a bug in the
    pdipe package. Please open an issue on https://github.com/pdpipe/pdpipe

    An example is a custom pipeline stage class that is an X-y transformer:
    meaning, it takes X and y as input and returns X and y. In order to declare
    itself as such, it implements the `_transform_Xy()` and the
    `_fit_transform_Xy` methods. However, since `_transform` is an abstract
    method of the PdPipeline class, it must be implemented, and so it is
    implemented to raise this error, expecting to never be called.

    Thus, if `_transform` is called instead of `_transform_Xy` for this stage,
    it is unexpected, and this exception is raised.
    """
