"""Classes for sklearn integration.

Despite similar names, there is a difference between pdpipe PdPipeline and
sklearn.pipeline.Pipeline. PdPipeline can only chain transformers while
scikit-learn Pipeline objects can further include the final estimator to
provide additional methods such as `predict` and `predict_proba`.

This means that by itself, pdpipe PdPipeline does not integrate well with some
of scikit-learn utility classes such as sklearn.model_selection.GridSearchCV
compared to sklearn.pipeline.Pipeline.

This module resolves such integration issues. Refer to the notebooks folder of
the pdpipe repository for complete examples.
"""

from typing import Callable
from functools import update_wrapper

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from .core import PdPipeline


def _estimator_has(attr):
    """Check if we can delegate a method to the underlying estimator.

    Calling a prediction method will only be available if `refit=True`. In
    such case, we check first the fitted best estimator. If it is not
    fitted, we check the unfitted estimator.

    Checking the unfitted estimator allows to use `hasattr` on the `SearchCV`
    instance even before calling `fit`.
    """

    def check(self):
        # raise an AttributeError if `attr` does not exist
        getattr(self.estimator, attr)
        return True

    return check


class _AvailableIfDescriptor:  # pragma: no cover
    """Implements a conditional property using the descriptor protocol.

    Using this class to create a decorator will raise an ``AttributeError``
    if check(self) returns a falsey value. Note that if check raises an error
    this will also result in hasattr returning false.

    See https://docs.python.org/3/howto/descriptor.html for an explanation of
    descriptors.
    """

    def __init__(self, fn, check, attribute_name):
        self.fn = fn
        self.check = check
        self.attribute_name = attribute_name

        # update the docstring of the descriptor
        update_wrapper(self, fn)

    def __get__(self, obj, owner=None):
        attr_err = AttributeError(
            f"This {repr(owner.__name__)} has no attribute "
            f"{repr(self.attribute_name)}"
        )
        if obj is not None:
            # delegate only on instances, not the classes.
            # this is to allow access to the docstrings.
            if not self.check(obj):
                raise attr_err

            # lambda, but not partial, allows help() to work with
            # update_wrapper
            out = lambda *args, **kwargs: self.fn(obj, *args, **kwargs)  # noqa
        else:

            def fn(*args, **kwargs):
                if not self.check(args[0]):
                    raise attr_err
                return self.fn(*args, **kwargs)

            # This makes it possible to use the decorated method as an
            # unbound method,
            # for instance when monkeypatching.
            out = lambda *args, **kwargs: fn(*args, **kwargs)  # noqa
        # update the docstring of the returned function
        update_wrapper(out, self.fn)
        return out


def available_if(check):
    """An attribute that is available only if check returns a truthy value.

    Parameters
    ----------
    check : callable
        When passed the object with the decorated method, this should return
        a truthy value if the attribute is available, and either return False
        or raise an AttributeError if not available.
    """
    return lambda fn: _AvailableIfDescriptor(
        fn, check, attribute_name=fn.__name__)


class PdPipelineAndSklearnEstimator(BaseEstimator):
    """A PdPipeline object chained before an sklearn estimator object.

    This kind of object can also be used with sklearn's GridSearchCV.

    See the pipeline_and_model.ipynb notebook in the notebooks folder of the
    pdpipe repository for a tutorial on how to use this class.

    Parameters
    ----------
    pipeline : PdPipeline
        The preprocssing pipeline to connect.
    model : sklearn.base.BaseEstimator
        The model to connect to the pipeline.

    Example
    ----------
        >>> import pandas as pd; import pdpipe as pdp;
        >>> from pdpipe.skintegrate import PdPipelineAndSklearnEstimator;
        >>> from sklearn.linear_model import LogisticRegression;
        >>> DF2 = pd.DataFrame(
        ...    data=[['-1',0], ['-1',0], ['1',1], ['1',1]],
        ...    index=[1, 2, 3, 4],
        ...    columns=['feature1', 'target']
        ... )
        >>> all_x = DF2[['feature1']]
        >>> all_y = DF2['target']
        >>> mp = PdPipelineAndSklearnEstimator(
        ...    pipeline=pdp.ColumnDtypeEnforcer({'feature1': int}),
        ...    estimator=LogisticRegression()
        ... )
        >>> mp.fit(all_x, all_y)
        <PdPipeline -> LogisticRegression>
        >>> res = mp.predict(all_x)
    """

    def __init__(
        self,
        pipeline: PdPipeline,
        estimator: BaseEstimator,
    ):
        self.pipeline = pipeline
        self.estimator = estimator
        # if hasattr(estimator, "score"):
        #     def _passthrough_scorer(estimator, *args, **kwargs):
        #         """Function that wraps estimator.score"""
        #         return estimator.score(*args, **kwargs)
        #     self.score = _passthrough_scorer

    def __str__(self):
        try:
            return f"<PdPipeline -> {self._est_cls_name}>"
        except AttributeError:
            self._est_cls_name = type(self.estimator).__name__
            return self.__str__()

    def __repr__(self):
        return self.__str__()

    def score(self, X, y=None):
        post_X = self.pipeline.transform(X)
        return self.estimator.score(post_X, y)

    @property
    def _estimator_type(self):
        return self.estimator._estimator_type

    @property
    def classes_(self):
        """Class labels.
        Only available when the estimator is a classifier.
        """
        _estimator_has("classes_")(self)
        return self.estimator.classes_

    def fit(self, X, y):
        """A reference implementation of a fitting function.

        Parameters
        ----------
        X : pandas.DataFrame, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        self : object
            Returns self.
        """
        # X, y = check_X_y(X, y, accept_sparse=True)
        post_X = self.pipeline.fit_transform(X=X, y=y)
        self.estimator.fit(X=post_X.values, y=y.values)
        self.is_fitted_ = True
        return self

    @available_if(_estimator_has("predict"))
    def predict(self, X):
        """ A reference implementation of a predicting function.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns an array of ones.
            The predicted labels or values for `X` based on the estimator with
            the best found parameters.
        """
        # X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')
        post_X = self.pipeline.transform(X=X)
        y_pred = self.estimator.predict(X=post_X.values)
        return y_pred

    @available_if(_estimator_has("predict_proba"))
    def predict_proba(self, X):
        """Call predict_proba on the estimator with the best found parameters.
        Only available if the underlying estimator supports
        ``predict_proba``.

        Parameters
        ----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.
        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_classes)
            Predicted class probabilities for `X` based on the estimator with
            the best found parameters. The order of the classes corresponds
            to that in the fitted attribute :term:`classes_`.
        """
        check_is_fitted(self, 'is_fitted_')
        post_X = self.pipeline.transform(X=X)
        y_pred = self.estimator.predict_proba(X=post_X.values)
        return y_pred

    @available_if(_estimator_has("predict_log_proba"))
    def predict_log_proba(self, X):
        """Call predict_log_proba on the estimator with the best found parameters.
        Only available if the underlying estimator supports
        ``predict_log_proba``.

        Parameters
        ----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.
        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_classes)
            Predicted class log-probabilities for `X` based on the estimator
            with the best found parameters. The order of the classes
            corresponds to that in the fitted attribute :term:`classes_`.
        """
        check_is_fitted(self, 'is_fitted_')
        post_X = self.pipeline.transform(X=X)
        y_pred = self.estimator.predict_log_proba(X=post_X.values)
        return y_pred

    @available_if(_estimator_has("decision_function"))
    def decision_function(self, X):
        """Call decision_function on the estimator with the best found parameters.
        Only available if the underlying estimator supports
        ``decision_function``.

        Parameters
        ----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.
        Returns
        -------
        y_score : ndarray of shape (n_samples,) or (n_samples, n_classes) \
                or (n_samples, n_classes * (n_classes-1) / 2)
            Result of the decision function for `X` based on the estimator with
            the best found parameters.
        """
        check_is_fitted(self, 'is_fitted_')
        post_X = self.pipeline.transform(X=X)
        y_score = self.estimator.decision_function(X=post_X.values)
        return y_score


# scorers that work with the pipline+model object

class _PdPipeScorer:
    """A pdpipe scorer object wrapping a standard sklearn scorer.

    Parameters
    ----------
    scorer : Callable
        The wrapped sklearn scorer.
    """

    def __init__(self, scorer: Callable) -> None:
        self._scorer = scorer

    def __call__(
        self,
        estimator: PdPipelineAndSklearnEstimator,
        X: pd.DataFrame,
        y=None,
        **kwargs,
    ):
        post_X = estimator.pipeline.transform(X)
        return self._scorer(
            estimator.estimator,
            post_X,
            y,
            **kwargs,
        )

    def __repr__(self) -> str:
        rs = repr(self._scorer)
        return f'<PdPipeScorer: {rs}>'


def pdpipe_scorer_from_sklearn_scorer(scorer: Callable) -> Callable:
    """Converts an sklearn scorer to one that will work with pdpipe.

    The returned scorer function can then be used with sklearn's
    model-evaluation tools using cross-validation (such as
    model_selection.cross_val_score and model_selection.GridSearchCV), when
    searching over the hyperparameter space of a PdPipelineAndSklearnEstimator
    object.

    See the pipeline_and_model_with_test_test.ipynb notebook in the notebooks
    folder of the pdpipe repository for a complete example.

    Parameters
    ----------
    scorer : callable
        A function with the signature `scorer(estimator, X, y)`. To build one
        from an sklearn `score` function (with a signature of the form
        `score(y_true, y_pred, ...)`) use the `sklearn.metrics.make_scorer`
        function.

    Returns
    -------
    pdpipe_scorer : callable
        A scorer that is aware of the fact that PdPipelineAndSklearnEstimator
        has an inner pipeline object that should be used to transform input
        X (which is a dataframe when using pdpipe, and not a numpy.ndarray).
    """
    return _PdPipeScorer(scorer)
