"""Setup for the pdpipe package."""

# !/usr/bin/env python
# -*- coding: utf-8 -*-

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

import versioneer


README_RST = ''
with open('README.rst', encoding="utf-8") as f:
    README_RST = f.read()

INSTALL_REQUIRES = [
    'pandas>=0.18.0',  # obviously
    'sortedcontainers',  # the Bin stage needs a sorted list
    'tqdm',  # for some pipeline application progress bars
    'strct',  # ColReorder uses strct.dicts.reverse_dict_partial
    'skutil>=0.0.15',  # Scale uses skutil.preprocessing.scaler_by_param
    'birch>=0.0.34',  # for reading config from files / env vars
]
TEST_REQUIRES = [
    # testing and coverage
    'pytest', 'coverage', 'pytest-cov', 'pytest-ordering',
    # non-testing packagesrequired by tests, not by the package
    'scikit-learn', 'pdutil', 'nltk', 'xdg',
    # to be able to run `python setup.py checkdocs`
    'collective.checkdocs', 'pygments',
]


setup(
    name='pdpipe',
    description="Easy pipelines for pandas.",
    long_description=README_RST,
    long_description_content_type='text/x-rst',
    author="Shay Palachy",
    author_email="shaypal5@gmail.com",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    url='https://pdpipe.github.io/pdpipe/',
    license="MIT",
    packages=['pdpipe'],
    install_requires=INSTALL_REQUIRES,
    extras_require={
        'sklearn': ['scikit-learn'],
        'nltk': ['nltk'],
        'test': TEST_REQUIRES
    },
    platforms=['any'],
    keywords='pandas dataframe pipeline data',
    classifiers=[
        # Trove classifiers
        # (https://pypi.python.org/pypi?%3Aaction=list_classifiers)
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Intended Audience :: Developers',
    ],
)
