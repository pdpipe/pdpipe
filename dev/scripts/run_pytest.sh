#!/bin/bash

# Run pytest with deprecation warnings ignored
python -m pytest -W ignore::DeprecationWarning "$@"
