"""Validate numpy docstrings throughout pdpipe."""

import re
import sys
import inspect
import subprocess

import pdpipe  # noqa: F401


# REGEXES

HIDE_ERROR_CODES_LIST = [
    "ES01",  # No extended summary found
    "SA01",  # See Also section not found
    "EX01",  # No examples section found
]

_COMPS = [":{}:".format(code) for code in HIDE_ERROR_CODES_LIST]
HIDE_ERROR_REGEX = "|".join(_COMPS)

HIDE_ERROR_PATTERN = re.compile(HIDE_ERROR_REGEX)

SOFT_ERROR_CODES_LIST = HIDE_ERROR_CODES_LIST + [
    # 'SS06',  # Summary should fit in a single line
    # 'GL01',  # Docstring text (summary) should start in the line immediately
    # after the opening quotes (not in the same line, or leaving a blank line
    # in between)
]

_COMPS = [":{}:".format(code) for code in SOFT_ERROR_CODES_LIST]
SOFT_ERROR_REGEX = "|".join(_COMPS)

SOFT_ERROR_PATTERN = re.compile(SOFT_ERROR_REGEX)


def get_npdoc_val_report(object_name) -> bool:
    """Prints a numpydoc validation report of the object of the given name.

    Parameters
    ----------
    object_name : str
        The name of the object for which to run numpydoc docstring validation.

    Returns
    -------
    bool
        True if any hard errors were found (errors not defined as soft errors
        in the SOFT_ERROR_CODES_LIST).

    """
    any_hard_errors = False
    try:
        output = subprocess.check_output(
            [
                "python",
                "-m",
                "numpydoc",
                object_name,
                "--validate",
            ]
        )
    except subprocess.CalledProcessError as e:
        output = e.output
    decoded_output = output.decode("utf-8")
    # if output includes ANY hard error code
    # calc by comparing N of soft errors w/ N lines
    report_lines = decoded_output.split("\n")
    nlines = len(report_lines) - 1
    nsoft = len(SOFT_ERROR_PATTERN.findall(decoded_output))
    nhide = len(HIDE_ERROR_PATTERN.findall(decoded_output))
    nerrors = nlines - nhide
    if nlines > nsoft:
        any_hard_errors = True
    if nerrors > 0:
        print(f"\nnumpydoc validation results for {object_name}:")
        print(f"A total of {nerrors} errors were found")
        print(
            f"Out of which {nsoft-nhide} are soft errors"
            f", (an additional {nhide} errors were hidden)."
        )
        for line in report_lines:
            if len(HIDE_ERROR_PATTERN.findall(line)) < 1 and len(line) > 0:
                print(line)
    return any_hard_errors


# list of types for which doc validation is skipped
NAME_SKIP = [
    "tqdm",
    "skintegrate",
    "count",
    "index",
    "Any",
    "Callable",
    "Optional",
    "List",
    "Tuple",
    "Set",
    "set",
    "Union",
    "Iterable",
    # pdpipe types that are not objects that should be checked
    "SeriesOperandTypesTuple",
    "ColumnsParamType",
    "ColumnLabelsType",
    # pdpipe globals we can skip
    "LOAD_STAGE_ATTRIBUTES",
    "POS_ARG_MISMTCH_PAT",
]


INSPECTED_CLASSES = []


def recursively_validate_object(val_obj, val_full_name, val_name) -> bool:
    """Recursively validate numpy docstrings of an object and its members."""
    if val_name in NAME_SKIP:
        return []
    val_cls_name = val_name[0].isupper()
    val_is_cls = inspect.isclass(val_obj) or val_cls_name
    if val_is_cls:
        if val_name in INSPECTED_CLASSES:
            return []
        INSPECTED_CLASSES.append(val_name)
    obj_w_hard_errors = []
    if get_npdoc_val_report(val_full_name):
        obj_w_hard_errors = [val_full_name]
    for name, obj in inspect.getmembers(val_obj):
        if not name.startswith("_"):
            cls_name = name[0].isupper()
            is_cls = inspect.isclass(obj) or cls_name
            is_func = inspect.isfunction(obj)
            if (is_cls and not val_is_cls) or (is_func and not is_cls):
                full_name = f"{val_full_name}.{name}"
                res = recursively_validate_object(obj, full_name, name)
                if len(res) > 0:
                    obj_w_hard_errors.extend(res)
    return sorted(set(obj_w_hard_errors))


module_blacklist = ["_version", "cfg"]


def validate_module(module_name: str) -> bool:
    """Validate numpy docstrings in an entire module.

    Parameters
    ----------
    module_name : str
        The name of the module for which to validate.

    Returns
    -------
    bool
        True if any hard errors were found; False otherwise.

    """
    print(f"Validating numpy docstrings in the {module_name} module!")
    obj_w_hard_errors = []
    res = get_npdoc_val_report(module_name)
    if res:
        obj_w_hard_errors.append(module_name)
    for name, obj in inspect.getmembers(sys.modules[module_name]):
        # print(obj)
        # if inspect.isclass(obj):
        #     print(obj)
        if inspect.ismodule(obj) and name not in module_blacklist:
            module_full_name = f"{module_name}.{name}"
            # print(f"Found pdpipe sub-module {module_full_name}")
            res = recursively_validate_object(obj, module_full_name, name)
            obj_w_hard_errors.extend(res)
    obj_w_hard_errors = sorted(set(obj_w_hard_errors))
    if len(obj_w_hard_errors) > 0:
        print("Hard errors were found in the following objects:")
        print("\n".join(obj_w_hard_errors))
        # print(obj_w_hard_errors)
    else:
        print("No hard errors were found anywhere in the module!")
    return len(obj_w_hard_errors) > 0


if __name__ == "__main__":
    errors_found = validate_module("pdpipe")
    if errors_found:
        sys.exit(1)
