import warnings

warnings.filterwarnings(
    "ignore",
    message=(
        ".*The `disp` and `iprint` options of the L-BFGS-B solver"
        " are deprecated.*"
    ),
    category=DeprecationWarning,
)
