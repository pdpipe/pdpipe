---
title: Easy pandas pipelines!
---
#
<p align="center">
    <img src="https://pdpipe.readthedocs.io/en/latest/images/pdpipe_row.png" alt="logowithtext" width="400px" style="display: block; margin-left: auto; margin-right: auto"/>
</p>


[![](https://img.shields.io/pypi/v/pdpipe.svg)](https://pypi.org/project/pdpipe)
[![](https://pepy.tech/badge/pdpipe)](https://pepy.tech/project/pdpipe)
[![](https://github.com/pdpipe/pdpipe/actions/workflows/test.yml/badge.svg)](https://github.com/pdpipe/pdpipe/actions/workflows/test.yml)
[![](https://codecov.io/github/pdpipe/pdpipe/coverage.svg?branch=master)](https://codecov.io/github/pdpipe/pdpipe?branch=master)
[![](https://www.codefactor.io/repository/github/pdpipe/pdpipe/badge?style=plastic)](https://www.codefactor.io/repository/github/pdpipe/pdpipe)
[![](https://img.shields.io/badge/License-MIT-ff69b4.svg)](https://pypi.python.org/pypi/pdpipe)
<!-- [![](https://img.shields.io/pypi/pyversions/pdpipe.svg)](https://pypi.org/project/pdpipe) -->


The `pdpipe` Python package provides a concise interface for building `pandas`
pipelines that have pre-conditions, are verbose, support the fit-transform
design of scikit-learn transformers and are highly serializable. `pdpipe`
pipelines have a simple interface, informative prints and errors on pipeline
application, support pipeline arithmetics and enable easier handling of
mixed-type data.

```py
>>> df = pd.DataFrame(
        data=[[4, 165, 'USA'], [2, 180, 'UK'], [2, 170, 'Greece']],
        index=['Dana', 'Jane', 'Nick'],
        columns=['Medals', 'Height', 'Born']
    )
>>> import pdpipe as pdp
>>> pipeline = pdp.ColDrop('Medals').OneHotEncode('Born')
>>> pipeline(df)
            Height  Born_UK  Born_USA
    Dana     165        0         1
    Jane     180        1         0
    Nick     170        0         0
```

## Installation

Install `pdpipe` using `pip`: `pip install pdpipe`

## Getting Help

* **Chat** — Join [our Gitter community](https://gitter.im/pdpipe/community) to chat for help!
* **Questions & Discussions** can be found on our [GitHub Discussions](https://github.com/pdpipe/pdpipe/discussions) forum.
* **Bugs and missing features** can opened on [our GitHub repository](https://github.com/pdpipe/pdpipe/issues).

## Getting Started

The awesome Tirthajyoti Sarkar wrote [an excellent practical introduction on how to use pdpipe](https://tirthajyoti.github.io/Notebooks/Pandas-pipeline-with-pdpipe).

For a thorough overview of all the capabilities of `pdpipe`, continue to the "Getting Started" section by clicking the "Next" on the bottom right.
