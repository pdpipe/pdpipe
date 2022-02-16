"""Test column bound potentials."""

import pytest
import pandas as pd
import numpy as np

import pdpipe as pdp
from pdpipe import df


def _num_df():
    return pd.DataFrame(
        data=[
            [-2, 2],
            [4, 3],
            [1, 1],
        ],
        columns=["a", "b"],
    )


def _pos_num_df():
    return pd.DataFrame(
        data=[
            [2, 2],
            [4, 3],
            [1, 1],
        ],
        columns=["a", "b"],
    )


def _bool_df():
    return pd.DataFrame(
        data=[
            [True, False],
            [True, True],
            [False, False],
            [False, True],
        ],
        columns=["a", "b"],
    )


def _num_df_with_na():
    return pd.DataFrame(
        data=[
            [-2, 2],
            [4, np.nan],
            [1, 1],
        ],
        columns=["a", "b"],
    )


class CustomType():
    """Custom type."""

    def __init__(self, a, b):
        self.a = a
        self.b = b


@pytest.mark.bound_col
def test_col_bound_potential_numerical_operators():
    """Test column bound potentials for numerical operators."""
    rdf = _num_df()
    c_vals = [8, 1, 3]
    d_val = 5
    # test unary operators
    pline = pdp.PdPipeline([
        df['-a'] << - df['a'],
        df['+a'] << + df['a'],
        df['abs(a)'] << abs(df['a']),
        df['c'] << pd.Series(c_vals),
        df['-c'] << -pd.Series(c_vals),
        df['d'] << d_val,
    ])
    res = pline(rdf)
    assert res['-a'].equals(-rdf['a'])
    assert res['+a'].equals(+rdf['a'])
    assert res['abs(a)'].equals(abs(rdf['a']))
    assert res['c'].equals(pd.Series(c_vals))
    assert res['-c'].equals(-pd.Series(c_vals))

    # test binary operators
    pline = pdp.PdPipeline([
        df['a<b'] << df['a'] < df['b'],
        df['a>b'] << df['a'] > df['b'],
        df['a<=b'] << df['a'] <= df['b'],
        df['a>=b'] << df['a'] >= df['b'],
        df['a==b'] << df['a'] == df['b'],
        df['a!=b'] << df['a'] != df['b'],
        df['a+b'] << df['a'] + df['b'],
        df['a-b'] << df['a'] - df['b'],
        df['a*b'] << df['a'] * df['b'],
        df['a/b'] << df['a'] / df['b'],
        df['a%b'] << df['a'] % df['b'],
        df['a**b'] << df['a'] ** df['b'],
        df['a//b'] << df['a'] // df['b'],
        # with c
        df['a<c'] << df['a'] < pd.Series(c_vals),
        df['a+d'] << df['a'] + d_val,
        # to the right
        (df['a'] + df['b']) >> df['a+b_right'],
    ])
    res = pline(rdf)
    assert res['a<b'].equals(rdf['a'] < rdf['b'])
    assert res['a>b'].equals(rdf['a'] > rdf['b'])
    assert res['a<=b'].equals(rdf['a'] <= rdf['b'])
    assert res['a>=b'].equals(rdf['a'] >= rdf['b'])
    assert res['a==b'].equals(rdf['a'].eq(rdf['b']))
    assert res['a!=b'].equals(rdf['a'] != rdf['b'])
    assert res['a+b'].equals(rdf['a'] + rdf['b'])
    assert res['a-b'].equals(rdf['a'] - rdf['b'])
    assert res['a*b'].equals(rdf['a'] * rdf['b'])
    assert res['a/b'].equals(rdf['a'] / rdf['b'])
    assert res['a%b'].equals(rdf['a'] % rdf['b'])
    assert res['a**b'].equals(rdf['a'] ** rdf['b'])
    assert res['a//b'].equals(rdf['a'] // rdf['b'])
    assert res['a<c'].equals(rdf['a'] < pd.Series(c_vals))
    assert res['a+d'].equals(rdf['a'] + d_val)
    assert res['a+b_right'].equals(rdf['a'] + rdf['b'])

    # test binary operators with SeriesFromDfAssigner
    pline = pdp.PdPipeline([
        (df['a+b'] << df['a']) + df['b'],
        (df['a-b'] << df['a']) - df['b'],
        (df['a*b'] << df['a']) * df['b'],
        (df['a/b'] << df['a']) / df['b'],
        (df['a%b'] << df['a']) % df['b'],
        (df['a**b'] << df['a']) ** df['b'],
        (df['a//b'] << df['a']) // df['b'],
        (df['a+d'] << df['a']) + d_val,
        df['a'] + (df['b'] >> df['a+b_right']),
    ])
    res = pline(rdf)
    assert res['a+b'].equals(rdf['a'] + rdf['b'])
    assert res['a-b'].equals(rdf['a'] - rdf['b'])
    assert res['a*b'].equals(rdf['a'] * rdf['b'])
    assert res['a/b'].equals(rdf['a'] / rdf['b'])
    assert res['a%b'].equals(rdf['a'] % rdf['b'])
    assert res['a**b'].equals(rdf['a'] ** rdf['b'])
    assert res['a//b'].equals(rdf['a'] // rdf['b'])
    assert res['a+d'].equals(rdf['a'] + d_val)
    assert res['a+b_right'].equals(rdf['a'] + rdf['b'])

    # test correct raising of TypeError
    custom = CustomType(1, 2)
    with pytest.raises(TypeError):
        df['a'] >> custom
    with pytest.raises(TypeError):
        df['a'] >> 4
    with pytest.raises(TypeError):
        pdp.PdPipeline([df['c'] << (df['a'] < custom)])
    with pytest.raises(TypeError):
        pdp.PdPipeline([df['c'] << (df['a'] > custom)])
    with pytest.raises(TypeError):
        pdp.PdPipeline([df['c'] << (df['a'] <= custom)])
    with pytest.raises(TypeError):
        pdp.PdPipeline([df['c'] << (df['a'] >= custom)])
    # with pytest.raises(TypeError):
    #     pdp.PdPipeline([df['c'] << (df['a'] == custom)])
    # with pytest.raises(TypeError):
    #     pdp.PdPipeline([df['c'] << (df['a'] != custom)])
    with pytest.raises(TypeError):
        pdp.PdPipeline([df['c'] << (df['a'] + custom)])
    with pytest.raises(TypeError):
        pdp.PdPipeline([df['c'] << (df['a'] - custom)])
    with pytest.raises(TypeError):
        pdp.PdPipeline([df['c'] << (df['a'] * custom)])
    with pytest.raises(TypeError):
        pdp.PdPipeline([df['c'] << (df['a'] / custom)])
    with pytest.raises(TypeError):
        pdp.PdPipeline([df['c'] << (df['a'] % custom)])
    with pytest.raises(TypeError):
        pdp.PdPipeline([df['c'] << (df['a'] ** custom)])
    with pytest.raises(TypeError):
        pdp.PdPipeline([df['c'] << (df['a'] // custom)])

    # test correct raising of TypeError for SeriesFromDfAssigner
    with pytest.raises(TypeError):
        pdp.PdPipeline([(df['a+b'] << df['a']) < custom])
    with pytest.raises(TypeError):
        pdp.PdPipeline([(df['a+b'] << df['a']) > custom])
    with pytest.raises(TypeError):
        pdp.PdPipeline([(df['a+b'] << df['a']) <= custom])
    with pytest.raises(TypeError):
        pdp.PdPipeline([(df['a+b'] << df['a']) >= custom])
    # with pytest.raises(TypeError):
    #     pdp.PdPipeline([(df['a+b'] << df['a']) == custom])
    # with pytest.raises(TypeError):
    #     pdp.PdPipeline([(df['a+b'] << df['a']) != custom])
    with pytest.raises(TypeError):
        pdp.PdPipeline([(df['c'] << df['a']) + custom])
    with pytest.raises(TypeError):
        pdp.PdPipeline([(df['c'] << df['a']) - custom])
    with pytest.raises(TypeError):
        pdp.PdPipeline([(df['c'] << df['a']) * custom])
    with pytest.raises(TypeError):
        pdp.PdPipeline([(df['c'] << df['a']) / custom])
    with pytest.raises(TypeError):
        pdp.PdPipeline([(df['c'] << df['a']) % custom])
    with pytest.raises(TypeError):
        pdp.PdPipeline([(df['c'] << df['a']) ** custom])
    with pytest.raises(TypeError):
        pdp.PdPipeline([(df['c'] << df['a']) // custom])


@pytest.mark.bound_col
def test_col_bound_potential_numerical_operators_with_series():
    """Test column bound potentials for numerical operators."""
    rdf = _num_df()
    c_vals = [8, 1, 3]
    cs = pd.Series(c_vals)

    # test binary operators with series
    pline = pdp.PdPipeline([
        df['a<cs'] << (df['a'] < cs),
        df['a>cs'] << (df['a'] > cs),
        df['a<=cs'] << (df['a'] <= cs),
        df['a>=cs'] << (df['a'] >= cs),
        df['a==cs'] << (df['a'] == cs),
        df['a!=cs'] << (df['a'] != cs),
        df['a+cs'] << (df['a'] + cs),
        df['a-cs'] << (df['a'] - cs),
        df['a*cs'] << (df['a'] * cs),
        df['a/cs'] << (df['a'] / cs),
        df['a%cs'] << (df['a'] % cs),
        df['a**cs'] << (df['a'] ** cs),
        df['a//cs'] << (df['a'] // cs),
    ])
    res = pline(rdf)
    assert res['a<cs'].equals(rdf['a'] < cs)
    assert res['a>cs'].equals(rdf['a'] > cs)
    assert res['a<=cs'].equals(rdf['a'] <= cs)
    assert res['a>=cs'].equals(rdf['a'] >= cs)
    assert res['a==cs'].equals(rdf['a'].eq(cs))
    assert res['a!=cs'].equals(rdf['a'] != cs)
    assert res['a+cs'].equals(rdf['a'] + cs)
    assert res['a-cs'].equals(rdf['a'] - cs)
    assert res['a*cs'].equals(rdf['a'] * cs)
    assert res['a/cs'].equals(rdf['a'] / cs)
    assert res['a%cs'].equals(rdf['a'] % cs)
    assert res['a**cs'].equals(rdf['a'] ** cs)
    assert res['a//cs'].equals(rdf['a'] // cs)

    # test binary operators with series for SeriesFromDfAssigner
    pline = pdp.PdPipeline([
        (df['a<cs'] << df['a']) < cs,
        (df['a>cs'] << df['a']) > cs,
        (df['a<=cs'] << df['a']) <= cs,
        (df['a>=cs'] << df['a']) >= cs,
        (df['a==cs'] << df['a']) == cs,
        (df['a!=cs'] << df['a']) != cs,
        (df['a+cs'] << df['a']) + cs,
        (df['a-cs'] << df['a']) - cs,
        (df['a*cs'] << df['a']) * cs,
        (df['a/cs'] << df['a']) / cs,
        (df['a%cs'] << df['a']) % cs,
        (df['a**cs'] << df['a']) ** cs,
        (df['a//cs'] << df['a']) // cs,
    ])
    res = pline(rdf)
    assert res['a<cs'].equals(rdf['a'] < cs)
    assert res['a>cs'].equals(rdf['a'] > cs)
    assert res['a<=cs'].equals(rdf['a'] <= cs)
    assert res['a>=cs'].equals(rdf['a'] >= cs)
    assert res['a==cs'].equals(rdf['a'].eq(cs))
    assert res['a!=cs'].equals(rdf['a'] != cs)
    assert res['a+cs'].equals(rdf['a'] + cs)
    assert res['a-cs'].equals(rdf['a'] - cs)
    assert res['a*cs'].equals(rdf['a'] * cs)
    assert res['a/cs'].equals(rdf['a'] / cs)
    assert res['a%cs'].equals(rdf['a'] % cs)
    assert res['a**cs'].equals(rdf['a'] ** cs)
    assert res['a//cs'].equals(rdf['a'] // cs)

    # test binary operators with series on the left
    pline = pdp.PdPipeline([
        df['cs<a'] << cs < df['a'],
        df['cs>a'] << cs > df['a'],
        df['cs<=a'] << cs <= df['a'],
        df['cs>=a'] << cs >= df['a'],
        df['cs==a'] << cs == df['a'],
        df['cs!=a'] << cs != df['a'],
        # these cases are not supported, since operator resolution order means
        # that the Series object on the left is enquired first for the
        # support of the __add__, __sub__, etc. methods for a
        # _BoundColumnPotential object, which they always support, interpreting
        # it as an object to add/sub/etc. element-wise. As such, the __radd__,
        # __rsub__, etc. methods of the _BoundColumnPotential object on the
        # right are never called, and we can't support the operation.
        # df['cs+a'] << cs + df['a'],
        # df['cs-a'] << cs - df['a'],
        # df['cs*a'] << cs * df['a'],
        # df['cs/a'] << cs / df['a'],
        # df['cs%a'] << cs % df['a'],
        # df['cs**a'] << cs ** df['a'],
        # df['cs//a'] << cs // df['a'],
    ])
    res = pline(rdf)
    assert res['cs<a'].equals(cs < rdf['a'])
    assert res['cs>a'].equals(cs > rdf['a'])
    assert res['cs<=a'].equals(cs <= rdf['a'])
    assert res['cs>=a'].equals(cs >= rdf['a'])
    assert res['cs==a'].equals(cs.eq(rdf['a']))
    assert res['cs!=a'].equals(cs != rdf['a'])
    # assert res['cs+a'].equals(cs + rdf['a'])
    # assert res['cs-a'].equals(cs - rdf['a'])
    # assert res['cs*a'].equals(cs * rdf['a'])
    # assert res['cs/a'].equals(cs / rdf['a'])
    # assert res['cs%a'].equals(cs % rdf['a'])
    # assert res['cs**a'].equals(cs ** rdf['a'])
    # assert res['cs//a'].equals(cs // rdf['a'])

    # test binary operators with a scalar on the left
    cs = 8
    pline = pdp.PdPipeline([
        df['cs<a'] << cs < df['a'],
        df['cs>a'] << cs > df['a'],
        df['cs<=a'] << cs <= df['a'],
        df['cs>=a'] << cs >= df['a'],
        df['cs==a'] << cs == df['a'],
        df['cs!=a'] << cs != df['a'],
        df['cs+a'] << cs + df['a'],
        df['cs-a'] << cs - df['a'],
        df['cs*a'] << cs * df['a'],
        df['cs/a'] << cs / df['a'],
        df['cs%a'] << cs % df['a'],
        df['cs**a'] << cs ** df['a'],
        df['cs//a'] << cs // df['a'],
    ])
    res = pline(rdf)
    assert res['cs<a'].equals(cs < rdf['a'])
    assert res['cs>a'].equals(cs > rdf['a'])
    assert res['cs<=a'].equals(cs <= rdf['a'])
    assert res['cs>=a'].equals(cs >= rdf['a'])
    assert res['cs==a'].equals(cs == rdf['a'])
    assert res['cs!=a'].equals(cs != rdf['a'])
    assert res['cs+a'].equals(cs + rdf['a'])
    # assert res['cs-a'].equals(cs - rdf['a'])
    # assert res['cs*a'].equals(cs * rdf['a'])
    # assert res['cs/a'].equals(cs / rdf['a'])
    # assert res['cs%a'].equals(cs % rdf['a'])

    # test binary operators with a scalar on the left
    rdf = _pos_num_df()
    cs = 8
    pline = pdp.PdPipeline([
        (df['cs+a'] << cs) + df['a'],
        (df['cs-a'] << cs) - df['a'],
        (df['cs*a'] << cs) * df['a'],
        (df['cs/a'] << cs) / df['a'],
        (df['cs%a'] << cs) % df['a'],
        (df['cs**a'] << cs) ** df['a'],
        (df['cs//a'] << cs) // df['a'],
    ])
    res = pline(rdf)
    assert res['cs+a'].equals(cs + rdf['a'])
    assert res['cs-a'].equals(cs - rdf['a'])
    assert res['cs*a'].equals(cs * rdf['a'])
    assert res['cs/a'].equals(cs / rdf['a'])
    assert res['cs%a'].equals(cs % rdf['a'])


@pytest.mark.bound_col
def test_col_bound_potential_boolean_operators():
    """Test column bound potentials for boolean operators."""
    rdf = _bool_df()
    # test unary operators
    pline = pdp.PdPipeline([
        df['not(a)'] << ~df['a'],
    ])
    res = pline(rdf)
    assert res['not(a)'].equals(~rdf['a'])

    # test binary operators
    pline = pdp.PdPipeline([
        df['a&b'] << df['a'] & df['b'],
        df['a|b'] << df['a'] | df['b'],
        df['a^b'] << df['a'] ^ df['b'],
        df['a&~b'] << df['a'] & ~df['b'],
        df['a|~b'] << df['a'] | ~df['b'],
        df['a^~b'] << df['a'] ^ ~df['b'],
    ])
    res = pline(rdf)
    assert res['a&b'].equals(rdf['a'] & rdf['b'])
    assert res['a|b'].equals(rdf['a'] | rdf['b'])
    assert res['a^b'].equals(rdf['a'] ^ rdf['b'])
    assert res['a&~b'].equals(rdf['a'] & ~rdf['b'])
    assert res['a|~b'].equals(rdf['a'] | ~rdf['b'])
    assert res['a^~b'].equals(rdf['a'] ^ ~rdf['b'])

    # test binary operators with series
    bool_series = pd.Series([True, False, True])
    pline = pdp.PdPipeline([
        df['a&b'] << df['a'] & bool_series,
        df['a|b'] << df['a'] | bool_series,
        df['a^b'] << df['a'] ^ bool_series,
        df['a&~b'] << df['a'] & ~bool_series,
        df['a|~b'] << df['a'] | ~bool_series,
        df['a^~b'] << df['a'] ^ ~bool_series,
    ])
    res = pline(rdf)
    assert res['a&b'].equals(rdf['a'] & bool_series)
    assert res['a|b'].equals(rdf['a'] | bool_series)
    assert res['a^b'].equals(rdf['a'] ^ bool_series)
    assert res['a&~b'].equals(rdf['a'] & ~bool_series)
    assert res['a|~b'].equals(rdf['a'] | ~bool_series)
    assert res['a^~b'].equals(rdf['a'] ^ ~bool_series)

    # test binary operators with series
    bool_series = pd.Series([True, False, True])
    pline = pdp.PdPipeline([
        df['a&b'] << (df['a'] & bool_series),
        df['a|b'] << (df['a'] | bool_series),
        df['a^b'] << (df['a'] ^ bool_series),
        df['a&~b'] << (df['a'] & ~bool_series),
        df['a|~b'] << (df['a'] | ~bool_series),
        df['a^~b'] << (df['a'] ^ ~bool_series),
    ])
    res = pline(rdf)
    assert res['a&b'].equals(rdf['a'] & bool_series)
    assert res['a|b'].equals(rdf['a'] | bool_series)
    assert res['a^b'].equals(rdf['a'] ^ bool_series)
    assert res['a&~b'].equals(rdf['a'] & ~bool_series)
    assert res['a|~b'].equals(rdf['a'] | ~bool_series)
    assert res['a^~b'].equals(rdf['a'] ^ ~bool_series)

    # test binary operators with seris on the left
    # === can't be supported for the same reason - pd.Series hijacks the binary
    # operator when on the left
    # maybe check this out one day:
    # https://pandas.pydata.org/docs/development/extending.html#extensionarray-operator-support
    # bool_series = pd.Series([True, False, True])
    # pline = pdp.PdPipeline([
    #     df['a&b'] << (bool_series & df['b']),
    #     df['a|b'] << (bool_series | df['b']),
    #     df['a^b'] << (bool_series ^ df['b']),
    #     df['a&~b'] << (bool_series & ~df['b']),
    #     df['a|~b'] << (bool_series | ~df['b']),
    #     df['a^~b'] << (bool_series ^ ~df['b']),
    # ])
    # res = pline(rdf)
    # assert res['a&b'].equals(bool_series & rdf['b'])
    # assert res['a|b'].equals(bool_series | rdf['b'])
    # assert res['a^b'].equals(bool_series ^ rdf['b'])
    # assert res['a&~b'].equals(bool_series & ~rdf['b'])
    # assert res['a|~b'].equals(bool_series | ~rdf['b'])
    # assert res['a^~b'].equals(bool_series ^ ~rdf['b'])

    # test binary operators with seris on the left for SeriesFromDfAssigner
    bool_series = pd.Series([True, False, True])
    pline = pdp.PdPipeline([
        df['a&b'] << bool_series & df['b'],
        df['a|b'] << bool_series | df['b'],
        df['a^b'] << bool_series ^ df['b'],
        df['a&~b'] << bool_series & ~df['b'],
        df['a|~b'] << bool_series | ~df['b'],
        df['a^~b'] << bool_series ^ ~df['b'],
    ])
    res = pline(rdf)
    assert res['a&b'].equals(bool_series & rdf['b'])
    assert res['a|b'].equals(bool_series | rdf['b'])
    assert res['a^b'].equals(bool_series ^ rdf['b'])
    assert res['a&~b'].equals(bool_series & ~rdf['b'])
    assert res['a|~b'].equals(bool_series | ~rdf['b'])
    assert res['a^~b'].equals(bool_series ^ ~rdf['b'])

    # test correct raising of TypeError
    custom = CustomType(1, 2)
    with pytest.raises(TypeError):
        pdp.PdPipeline([df['c'] << (df['a'] & custom)])
    with pytest.raises(TypeError):
        pdp.PdPipeline([df['c'] << (df['a'] | custom)])
    with pytest.raises(TypeError):
        pdp.PdPipeline([df['c'] << (df['a'] ^ custom)])

    # test correct raising of TypeError for SeriesFromDfAssigner
    with pytest.raises(TypeError):
        pdp.PdPipeline([df['c'] << df['a'] & custom])
    with pytest.raises(TypeError):
        pdp.PdPipeline([df['c'] << df['a'] | custom])
    with pytest.raises(TypeError):
        pdp.PdPipeline([df['c'] << df['a'] ^ custom])


@pytest.mark.bound_col
def test_col_bound_potential_complex():
    """Test column bound potentials for boolean operators."""
    rdf = _num_df_with_na()
    c_vals = [8, 1, 3]
    c_s = pd.Series(c_vals)
    d_val = 5
    # test unary operators
    pline = pdp.PdPipeline([
        df['-a'] << - df['a'].isin([3, 4, 5]),
        df['filled_b'] << df['b'].fillna(0),
    ])
    res = pline(rdf)
    assert res['-a'].equals(-rdf['a'].isin([3, 4, 5]))
    assert res['filled_b'].equals(rdf['b'].fillna(0))

    # test binary operators
    MAP = {-2: 7, 1: 5, 4: 1}
    pline = pdp.PdPipeline([
        df['a_map'] << df['a'].map(MAP),
        df['a<b'] << df['a'].map(MAP) < df['b'].fillna(1),
        df['a_map>b+d'] << df['a'].map(MAP) > df['b'] + d_val,
        df['a_map+d'] << df['a'].map(MAP) + d_val,
        df['a_map>d+b'] << df['a'].map(MAP) > d_val + df['b'],
        df['a_map+b+c'] << df['a'].map(MAP) + df['b'] + c_s,
        df['a_add_b'] << df['a'].add(df['b'], fill_value=0),
        df['a_add_b_kwarg'] << df['a'].add(other=df['b'], fill_value=0),
    ])
    res = pline(rdf)
    assert res['a_map'].equals(rdf['a'].map(MAP))
    assert res['a<b'].equals(rdf['a'].map(MAP) < rdf['b'].fillna(1))
    assert res['a_map>b+d'].equals(rdf['a'].map(MAP) > rdf['b'] + d_val)
    assert res['a_map+d'].equals(rdf['a'].map(MAP) + d_val)
    assert res['a_map>d+b'].equals(rdf['a'].map(MAP) > d_val + rdf['b'])
    assert res['a_map+b+c'].equals(rdf['a'].map(MAP) + rdf['b'] + c_s)
    assert res['a_add_b'].equals(rdf['a'].add(rdf['b'], fill_value=0))
    assert res['a_add_b_kwarg'].equals(
        rdf['a'].add(other=rdf['b'], fill_value=0))
