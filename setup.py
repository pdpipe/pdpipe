"""Setup for the pdpipe package."""

# !/usr/bin/env python
# -*- coding: utf-8 -*-

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

import versioneer


README_RST = ''
with open('README.rst') as f:
    README_RST = f.read()

INSTALL_REQUIRES = [
    'pandas>=0.18.0', 'sortedcontainers', 'tqdm',
]
TEST_REQUIRES = ['pytest', 'coverage', 'pytest-cov']


setup(
    name='pdpipe',
    description="Easy pipelines for pandas.",
    long_description=README_RST,
    author="Shay Palachy",
    author_email="shaypal5@gmail.com",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    url='https://github.com/shaypal5/pdpipe',
    license="MIT",
    packages=['pdpipe'],
    install_requires=INSTALL_REQUIRES,
    extras_require={
        'sklearn': ['scikit-learn', 'skutil'],
        'nltk': ['nltk'],
        'test': TEST_REQUIRES
    },
    setup_requires=INSTALL_REQUIRES,
    platforms=['any'],
    keywords='pandas dataframe pipeline data',
    classifiers=[
        # Trove classifiers
        # (https://pypi.python.org/pypi?%3Aaction=list_classifiers)
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Intended Audience :: Developers',
    ],
)
