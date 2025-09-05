"""Custom exceptions for pdpipe."""


class FailedPreconditionError(Exception):
    """Raised when a stage is applied to a dataframe violating its
    precondition."""


class FailedPostconditionError(Exception):
    """Raised when an expected post-condition is violated after stage
    application."""


class FailedConditionError(Exception):
    """Raised when an expected condition does not hold for an input
    dataframe."""


class PipelineInitializationError(Exception):
    """An exception raised when a pipeline is not initialized properly."""


class PipelineApplicationError(Exception):
    """Raised when pipeline application raises an error."""


class UnfittedPipelineStageError(Exception):
    """Raised when a transform is attempted with an unfitted pipeline stage."""


class UnexpectedPipelineMethodCallError(Exception):
    """Raised a placeholder method implementation is called unexpectedly.

    An exception raised when a placeholder method implementation of an
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
