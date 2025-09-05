# GitHub Copilot Custom Instructions for pdpipe

- pdpipe is a Python package for building serializable, chainable, and verbose data processing pipelines for pandas DataFrames, with a focus on data science and machine learning workflows.
- Always follow the fit-transform design pattern compatible with scikit-learn transformers when discussing pipeline stages or API design.
- Code must adhere to flake8 and black formatting standards (see pyproject.toml for details). Linting is enforced in CI.
- Use numpy docstring conventions for all public functions, classes, and methods. Include clear parameter and return type documentation and examples where possible.
- Do not mutate input DataFrames in place; all transformations should return new DataFrames.
- When adding new pipeline stages, use informative, explicit naming (e.g., ColDrop, ValDrop) to maximize pipeline readability.
- Tests must be added for all new code. Place tests in the appropriate module subdirectory under tests/, and use one test function per use case. Aim to maintain 100% test coverage.
- Doctests and code examples are encouraged in docstrings for new stages and functions.
- Some features are optional and require scikit-learn or nltk; code should gracefully degrade if these are not installed, issuing a warning but not failing.
- Pipelines and stages should be highly configurable and support serialization/deserialization for production use.
- Default behaviors should help users avoid common data science pitfalls (e.g., one-hot encoding drops one column by default to avoid the dummy variable trap).
- pdpipe supports Python 3.9 and up. Ensure compatibility with all supported versions.
- For configuration, support both config files and environment variables as described in the README.
- For help, reference the official documentation at https://pdpipe.readthedocs.io/en/latest/ and the Gitter community.
