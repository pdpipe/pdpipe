"""Testing basic pipeline stages."""
from builtins import ValueError

import pandas as pd
import pytest

from pdpipe.core import (
    PdPipelineStage,
    PdPipeline
)
from pdpipe import make_pdpipeline, ColByFrameFunc, ColDrop
from pdpipe.exceptions import PipelineApplicationError


def _test_df():
    return pd.DataFrame(
        data=[[1, 2, 'a'], [2, 4, 'b']],
        index=[1, 2],
        columns=['num1', 'num2', 'char']
    )


class SilentDropStage(PdPipelineStage):
    """A pipeline stage for testing"""

    def __init__(self, colname, **kwargs):
        self.colname = colname
        super().__init__(exraise=False, **kwargs)

    def _prec(self, df):
        return self.colname in df.columns

    def _transform(self, df, verbose):
        return df.drop([self.colname], axis=1)


@pytest.mark.parametrize("time", [True, False])
def test_two_stage_pipeline_stage(time):
    """Testing something."""
    drop_num1 = SilentDropStage('num1')
    drop_num2 = SilentDropStage('num2')
    pipeline = PdPipeline([drop_num1, drop_num2])
    assert len(pipeline) == 2
    df = _test_df()
    res_df = pipeline.apply(df, verbose=True)
    assert 'num1' not in res_df.columns
    assert 'num2' not in res_df.columns
    assert 'char' in res_df.columns
    str(pipeline)
    pipeline.memory_report()

    # test fit
    df = _test_df()
    res_df = pipeline.fit(df, verbose=True, time=time)
    for x in ['num1', 'num2', 'char']:
        assert x in res_df.columns

    # test apply
    df = _test_df()
    res_df = pipeline.apply(df, verbose=True, time=time)
    assert 'num1' not in res_df.columns
    assert 'num2' not in res_df.columns
    assert 'char' in res_df.columns

    # test transform
    df = _test_df()
    res_df = pipeline.transform(df, verbose=True, time=time)
    assert 'num1' not in res_df.columns
    assert 'num2' not in res_df.columns
    assert 'char' in res_df.columns

    # test fit_transform
    df = _test_df()
    res_df = pipeline.fit_transform(df, verbose=True, time=time)
    assert 'num1' not in res_df.columns
    assert 'num2' not in res_df.columns
    assert 'char' in res_df.columns

    # test get_transformer
    trs = lambda pipline: pipeline[:1]  # noqa: E731
    pipeline = PdPipeline([drop_num1, drop_num2], transformer_getter=trs)
    transformer = pipeline.get_transformer()
    res_df = transformer(df, verbose=True)
    assert 'num1' not in res_df.columns
    assert 'num2' in res_df.columns
    assert 'char' in res_df.columns


def test_make_pdpipeline():
    """Testing something."""
    drop_num1 = SilentDropStage('num1')
    drop_num2 = SilentDropStage('num2')
    pipeline = make_pdpipeline(drop_num1, drop_num2)
    assert len(pipeline) == 2
    df = _test_df()
    res_df = pipeline.apply(df, verbose=True)
    assert 'num1' not in res_df.columns
    assert 'num2' not in res_df.columns
    assert 'char' in res_df.columns


def test_pipeline_stage_addition():
    """Testing something."""
    drop_num1 = SilentDropStage('num1')
    drop_num2 = SilentDropStage('num2')
    pipeline = drop_num1 + drop_num2
    assert len(pipeline) == 2
    df = _test_df()
    res_df = pipeline.apply(df, verbose=True)
    assert 'num1' not in res_df.columns
    assert 'num2' not in res_df.columns
    assert 'char' in res_df.columns


def test_pipeline_to_pipeline_stage_addition():
    """Testing something."""
    drop_num1 = SilentDropStage('num1')
    drop_num2 = SilentDropStage('num2')
    pipeline = PdPipeline([drop_num1])
    assert len(pipeline) == 1
    pipeline = pipeline + drop_num2
    assert len(pipeline) == 2
    df = _test_df()
    res_df = pipeline.apply(df, verbose=True)
    assert 'num1' not in res_df.columns
    assert 'num2' not in res_df.columns
    assert 'char' in res_df.columns


def test_pipeline_stage_to_pipeline_addition():
    """Testing something."""
    drop_num1 = SilentDropStage('num1')
    drop_num2 = SilentDropStage('num2')
    pipeline = PdPipeline([drop_num1])
    assert len(pipeline) == 1
    pipeline = drop_num2 + pipeline
    assert len(pipeline) == 2
    df = _test_df()
    res_df = pipeline.apply(df, verbose=True)
    assert 'num1' not in res_df.columns
    assert 'num2' not in res_df.columns
    assert 'char' in res_df.columns


def test_pipeline_to_pipeline_addition():
    """Testing something."""
    drop_num1 = SilentDropStage('num1')
    drop_num2 = SilentDropStage('num2')
    pipeline1 = PdPipeline([drop_num1])
    pipeline2 = PdPipeline([drop_num2])
    pipeline = pipeline1 + pipeline2
    assert len(pipeline) == 2
    assert pipeline[0] == drop_num1
    assert pipeline[1] == drop_num2
    df = _test_df()
    res_df = pipeline.apply(df, verbose=True)
    assert 'num1' not in res_df.columns
    assert 'num2' not in res_df.columns
    assert 'char' in res_df.columns


def test_pipeline_to_int_addition():
    """Testing something."""
    pipeline = PdPipeline([SilentDropStage('num1')])
    with pytest.raises(TypeError):
        res = pipeline + 43
        assert not isinstance(res, PdPipeline)


def test_pipeline_index():
    """Testing something."""
    df = _test_df()
    drop_num1 = SilentDropStage('num1')
    drop_num2 = SilentDropStage('num2')
    drop_char = SilentDropStage('char')
    pipeline = PdPipeline([drop_num1, drop_num2, drop_char])
    assert len(pipeline) == 3
    assert pipeline[0] == drop_num1
    assert 'num1' not in pipeline[0](df).columns
    assert pipeline[1] == drop_num2
    assert 'num2' not in pipeline[1](df).columns
    assert pipeline[2] == drop_char
    assert 'char' not in pipeline[2](df).columns


def test_pipeline_slice():
    """Testing something."""
    drop_num1 = SilentDropStage('num1')
    drop_num2 = SilentDropStage('num2')
    drop_char = SilentDropStage('char')
    pipeline = PdPipeline([drop_num1, drop_num2, drop_char])
    assert len(pipeline) == 3
    pipeline = pipeline[0:2]
    assert len(pipeline) == 2
    df = _test_df()
    res_df = pipeline.apply(df, verbose=True)
    assert 'num1' not in res_df.columns
    assert 'num2' not in res_df.columns
    assert 'char' in res_df.columns


def test_pipeline_slice_by_name():
    """Testing something."""
    drop_num1 = SilentDropStage('num1', name='dropNum1')
    drop_num2 = SilentDropStage('num2', name='dropNum2')
    drop_char = SilentDropStage('char', name='dropChar')
    pipeline = PdPipeline([drop_num1, drop_num2, drop_char])
    assert len(pipeline) == 3
    pipeline = pipeline[['dropNum1', 'dropNum2']]
    assert len(pipeline) == 2
    assert pipeline['dropNum1'] == drop_num1
    with pytest.raises(ValueError) as e:
        pipeline['dropChar']
    assert str(e.value) == "'dropChar' is not exist."
    df = _test_df()
    res_df = pipeline.apply(df, verbose=True)
    assert 'num1' not in res_df.columns
    assert 'num2' not in res_df.columns
    assert 'char' in res_df.columns


@pytest.mark.parametrize("time", [True, False])
def test_pipeline_error(time):
    """Test exceptions at pipeline level"""

    # test fit
    df = _test_df()

    def _func(df):
        return df['num1'] == df['num3']
    pipeline = PdPipeline([ColByFrameFunc("Equality", _func), ColDrop("B")])
    with pytest.raises(PipelineApplicationError):
        pipeline.fit(df, verbose=True, time=time)

    # test apply
    df = _test_df()
    with pytest.raises(PipelineApplicationError):
        pipeline.apply(df, verbose=True, time=time)

    # test transform
    df = _test_df()
    with pytest.raises(PipelineApplicationError):
        pipeline.transform(df, verbose=True, time=time)

    # test fit_transform
    df = _test_df()
    with pytest.raises(PipelineApplicationError):
        pipeline.fit_transform(df, verbose=True, time=time)
