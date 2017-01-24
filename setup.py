"""Setup for the pandlines module."""

#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import os

from setuptools import setup, find_packages
import versioneer

with open('README.rst') as f:
    README = f.read()

setup(
    author="Shay Palachy",
    author_email="shaypal5@gmail.com",
    name='pandlines',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    long_description=README,
    url='https://github.com/shaypal5/pandlines',
    packages=find_packages(),
    install_requires=[
        'pandas'
    ],
    test_suite='nose.collector',
    tests_require=['nose'],
)
