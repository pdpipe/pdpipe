"""Testing the ApplicationContextEnricher stage."""

import pytest
import pandas as pd

from pdpipe import (
    PdPipelineStage,
    PdPipeline,
)
from pdpipe.basic_stages import ApplicationContextEnricher
from pdpipe.exceptions import PipelineApplicationError


DF1 = pd.DataFrame({'a': ['a', 'b', 'c', 'd'], 'b': [5, 6, 7, 1]})


class ApplicationContextAsserter(PdPipelineStage):

    def __init__(self, **kwargs):
        self._enrichments = kwargs
        super_kwargs = {
            'desc': "Assert application context enrichments"
        }
        super().__init__(**super_kwargs)

    def _prec(self, df):
        return True

    def _transform(self, df, verbose):
        for k, v in self._enrichments.items():
            assert self.application_context[k] == v


def test_application_context_basic():
    df1 = DF1
    pline = PdPipeline([
        ApplicationContextEnricher(
            bsum=lambda df: df['b'].sum(),
            bmean=lambda df: df['b'].mean(),
            bdiff=lambda df, application_context:
                application_context['bsum'] - application_context['bmean'],
            d=5,
        ),
        ApplicationContextAsserter(
            bsum=df1['b'].sum(),
            bmean=df1['b'].mean(),
            bdiff=df1['b'].sum() - df1['b'].mean(),
            d=5,
        ),
    ])
    pline(df1)


def test_application_context_error():
    df1 = DF1
    pline = PdPipeline([
        ApplicationContextEnricher(
            asum=lambda df: (2 + df['a']).sum(),
        ),
    ])
    with pytest.raises(PipelineApplicationError):
        pline(df1)
