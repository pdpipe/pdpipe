"""Can help expand series transforms whitelist and blacklist occasionally."""


if __name__ == '__main__':
    from pandas import Series
    from pdpipe.df.series_transformer import _has_series_transform_doc
    from rich.console import Console
    console = Console()
    print = console.print
    ser = Series([1, 2, 3])
    transforms = []
    not_transforms = []
    for attr_name in dir(Series):
        attr = getattr(Series, attr_name)
        current_has_series_transform_doc = False
        if _has_series_transform_doc(attr_name, attr):
            # print(f"Potential series transform: {attr_name} of {attr}")
            method = getattr(ser, attr_name)
            try:
                res = method()
                if isinstance(res, Series):
                    current_has_series_transform_doc = True
            except Exception:
                pass
            try:
                res = method(1)
                if isinstance(res, Series):
                    current_has_series_transform_doc = True
            except Exception:
                pass
            try:
                res = method(ser)
                if isinstance(res, Series):
                    current_has_series_transform_doc = True
            except Exception:
                pass
            if current_has_series_transform_doc:
                transforms.append(attr_name)
                print(
                    f"[blue bold]\t{attr_name}[/blue bold] [green]is a "
                    "series transform[/green]")
            else:
                not_transforms.append(attr_name)
                print(
                    f"[blue bold]\t{attr_name}[/blue bold] [red]is not a "
                    "series transform[/red]")
    print("\n[blue bold]Transforms[/blue bold]")
    print(f"{transforms}")
    print("\n[blue bold]Potential Non-Transforms[/blue bold]")
    print(f"{not_transforms}")
