"""Testing the ApplicationContextEnricher stage."""

import pickle

import pandas as pd
import pytest

from pdpipe import (
    PdPipeline,
    PdPipelineStage,
)
from pdpipe.basic_stages import ApplicationContextEnricher
from pdpipe.exceptions import PipelineApplicationError

from pdptestutil import random_pickle_path

DF1 = pd.DataFrame({"a": ["a", "b", "c", "d"], "b": [5, 6, 7, 1]})


class ApplicationContextAsserter(PdPipelineStage):
    def __init__(self, **kwargs):
        self._enrichments = kwargs
        super_kwargs = {"desc": "Assert application context enrichments"}
        super().__init__(**super_kwargs)

    def _prec(self, df):
        return True

    def _transform(self, df, verbose):
        for k, v in self._enrichments.items():
            assert self.application_context[k] == v


def test_application_context_basic():
    df1 = DF1
    pline = PdPipeline(
        [
            ApplicationContextEnricher(
                bsum=lambda df: df["b"].sum(),
                bmean=lambda df: df["b"].mean(),
                bdiff=lambda df, application_context: application_context[
                    "bsum"
                ]
                - application_context["bmean"],
                d=5,
            ),
            ApplicationContextAsserter(
                bsum=df1["b"].sum(),
                bmean=df1["b"].mean(),
                bdiff=df1["b"].sum() - df1["b"].mean(),
                d=5,
            ),
        ]
    )
    pline(df1)


def test_application_context_error():
    df1 = DF1
    pline = PdPipeline(
        [
            ApplicationContextEnricher(
                asum=lambda df: (2 + df["a"]).sum(),
            ),
        ]
    )
    with pytest.raises(PipelineApplicationError):
        pline(df1)


def _bsum(df):
    return df["b"].sum()


def _bmean(df):
    return df["b"].mean()


def test_pickle_app_context_enricher(pdpipe_tests_dir_path):
    """Testing ApplicationContextEnricher pickling."""
    stage = ApplicationContextEnricher(bsum=_bsum, bmean=_bmean, d=5)
    fpath = random_pickle_path(pdpipe_tests_dir_path)
    with open(fpath, "wb+") as f:
        pickle.dump(stage, f)
    with open(fpath, "rb") as f:
        loaded_stage = pickle.load(f)
    pline = PdPipeline(
        [
            loaded_stage,
            ApplicationContextAsserter(
                bsum=DF1["b"].sum(),
                bmean=DF1["b"].mean(),
                d=5,
            ),
        ]
    )
    pline(DF1)
