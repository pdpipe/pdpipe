
# Installation

Install `pdpipe` with:

```bash
  pip install pdpipe
```

## Optional Requirements

Some pipeline stages require `scikit-learn`; they will simply not be loaded if `scikit-learn` is not found on the system, and `pdpipe` will issue a warning. To use them you must also [install scikit-learn](http://scikit-learn.org/stable/install.html).


Similarly, some pipeline stages require `nltk`; they will not be loaded if `nltk` is not found on your system, and `pdpipe` will issue a warning. To use them you must additionally [install nltk](http://www.nltk.org/install.html).

