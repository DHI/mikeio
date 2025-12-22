"""Utilities for deprecating function arguments."""

from __future__ import annotations

import functools
import inspect
import warnings
from typing import Callable, TypeVar

F = TypeVar("F", bound=Callable)


def _deprecate_positional_args(
    func: F | None = None, *, start_after: str | None = None
) -> F | Callable[[F], F]:
    """Decorator to deprecate positional arguments.

    This decorator warns users when they pass arguments positionally that will
    become keyword-only in a future version.

    Based on scikit-learn's approach (SLEP009).

    Parameters
    ----------
    func : callable, optional
        Function to wrap. If None, returns a decorator with start_after parameter.
    start_after : str, optional
        Name of the last parameter that should remain positional.
        All parameters after this one will require keyword usage.
        If None, looks for parameters after * in the function signature.

    Returns
    -------
    callable
        Wrapped function that issues warnings for positional arguments

    Examples
    --------
    >>> @_deprecate_positional_args
    ... def scale(infile, outfile, *, offset=0.0, factor=1.0):
    ...     pass
    >>> scale("in.dfs", "out.dfs", 5.0)  # Issues FutureWarning

    >>> @_deprecate_positional_args(start_after="outfile")
    ... def scale(infile, outfile, offset=0.0, factor=1.0):
    ...     pass
    >>> scale("in.dfs", "out.dfs", 5.0)  # Issues FutureWarning

    """

    def decorator(f: F) -> F:
        sig = inspect.signature(f)
        params_list = list(sig.parameters.items())

        # Find parameters that will become keyword-only
        if start_after is not None:
            # Find the position after start_after parameter
            positional_params = []
            future_kwonly = []
            found_start_after = False

            for param_name, param in params_list:
                if param_name == start_after:
                    positional_params.append(param_name)
                    found_start_after = True
                elif not found_start_after:
                    positional_params.append(param_name)
                else:
                    # Parameters after start_after will become keyword-only
                    if param.kind in (
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        inspect.Parameter.POSITIONAL_ONLY,
                    ):
                        future_kwonly.append(param_name)

            if not found_start_after:
                raise ValueError(
                    f"Parameter '{start_after}' not found in function signature"
                )
        else:
            # Look for existing keyword-only parameters (after * in signature)
            future_kwonly = []
            positional_params = []
            found_kwonly_sep = False

            for param_name, param in params_list:
                if param.kind == inspect.Parameter.VAR_POSITIONAL:
                    found_kwonly_sep = True
                elif param.kind == inspect.Parameter.KEYWORD_ONLY:
                    future_kwonly.append(param_name)
                    found_kwonly_sep = True
                elif not found_kwonly_sep:
                    positional_params.append(param_name)

        # If no future keyword-only args, nothing to deprecate
        if not future_kwonly:
            return f

        # Number of allowed positional arguments
        num_positional_allowed = len(positional_params)

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            # Check if too many positional arguments were passed
            num_args_passed = len(args)

            if num_args_passed > num_positional_allowed:
                # Calculate how many extra positional args were passed
                num_extra = num_args_passed - num_positional_allowed

                # Identify which future keyword-only args were passed positionally
                args_passed_positionally = []
                for i in range(num_extra):
                    if i < len(future_kwonly):
                        param_name = future_kwonly[i]
                        param_value = args[num_positional_allowed + i]
                        args_passed_positionally.append((param_name, param_value))

                if args_passed_positionally:
                    args_msg = ", ".join(
                        f"{name}={value!r}" for name, value in args_passed_positionally
                    )

                    warnings.warn(
                        f"Passing {args_msg} as positional argument(s) is "
                        f"deprecated since version 3.1 and will raise an error in version 4.0. "
                        f"Please use keyword argument(s) instead.",
                        FutureWarning,
                        stacklevel=2,
                    )

            return f(*args, **kwargs)

        return wrapper

    # Handle both @decorator and @decorator(...) syntax
    if func is None:
        # Called with arguments: @_deprecate_positional_args(start_after="x")
        return decorator
    else:
        # Called without arguments: @_deprecate_positional_args
        return decorator(func)
