import contextlib
import logging
import os
import io
import subprocess
from contextlib import redirect_stdout, redirect_stderr
from typing import List, Any, Optional, Dict, Callable

from typeguard import TypeCheckError

from mokapot.mokapot import main as mokapot_main


@contextlib.contextmanager
def catch_type_check():
    """
    Catch Type Check

    Context manager that catches TypeCheckError exceptions and prints detailed
    error information.

    Usage:
    ------
    with catch_type_check():
        # code that may raise TypeCheckError
    """
    try:
        yield None
    except TypeCheckError as e:
        import traceback

        logging.error("\n\nTypeCheckError: ", end="")
        # Print the exception with the stack trace excluding the last
        # entries which stem from the typeguard module
        frames = traceback.extract_tb(e.__traceback__)
        entries_to_print = sum(
            1 for frame in frames if "typeguard" not in frame.filename
        )
        traceback.print_exception(e, limit=entries_to_print)
        raise


def _run_cli(
    module: str,
    main_func: Callable,
    params: List[Any],
    run_in_subprocess: Optional[bool] = None,
    capture_output: bool = False,
) -> Optional[Dict[str, str]]:
    """
    Parameters
    ----------
    module : str
        The name of the module to run as a script.
    main_func : Callable
        The main function to execute within the module.
    params : List[Any]
        The list of parameters to pass to the main function.
    run_in_subprocess : Optional[bool], default: None
        Determines whether to run the module in a subprocess. If `None`, the
        value is determined by the environment
        variable `TEST_CLI_IN_SUBPROCESS`
    capture_output : bool, default: False
        Determines whether to capture the stdout and stderr output.

    Returns
    -------
    Optional[Dict[str, str]]
        Returns a dictionary with the captured stdout and stderr if
        `capture_output` is True. Otherwise, returns None.
    """
    if run_in_subprocess is None:
        value = os.environ.get("TEST_CLI_IN_SUBPROCESS", "true")
        run_in_subprocess = value.lower() in ["true", "yes", "y", "1"]
    else:
        run_in_subprocess = bool(run_in_subprocess)

    params = [str(param) for param in params]
    if run_in_subprocess:
        cmd = ["python", "-m", module] + params
        try:
            res = subprocess.run(
                cmd, check=True, capture_output=capture_output
            )
        except subprocess.CalledProcessError as e:
            logging.error(
                f"Calling {module} with params {params} failed"
                f"\nStderr: {e.stderr}"
            )
            raise RuntimeError(
                f"Calling {module} with params {params} failed"
            ) from e
        if capture_output:
            return {
                "stdout": res.stdout.decode(),
                "stderr": res.stderr.decode(),
            }
    elif capture_output:
        stdout_sink = io.StringIO()
        stderr_sink = io.StringIO()
        logger = logging.getLogger()

        handler = logging.StreamHandler(stderr_sink)
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        with redirect_stderr(stderr_sink), redirect_stdout(stdout_sink):
            with catch_type_check():
                main_func(params)
        return {
            "stdout": stdout_sink.getvalue(),
            "stderr": stderr_sink.getvalue(),
        }
    else:
        with catch_type_check():
            main_func(params)


def run_mokapot_cli(
    params: List[Any], run_in_subprocess=None, capture_output=False
):
    """
    Run the mokapot command either in a subprocess or as direct
    call to the main function.

    Parameters
    ----------
    params : List[Any]
        List of parameters to be passed to the `mokapot.aggregatePsmsToPeptides`
        command line tool.

    run_in_subprocess : bool, optional
        If set to True, the script will be run in a subprocess. If set to False,
        the script will be run directly. If not provided, the value will be read
        from the environment variable TEST_CLI_IN_SUBPROCESS. If the value is
        'true', 'yes', 'y', or '1', it will be treated as True. Otherwise, it
        will be treated as False. Default is None.

    capture_output : bool, default: False
        Determines whether to capture the stdout and stderr output.
    """  # noqa: E501
    return _run_cli(
        "mokapot.mokapot",
        mokapot_main,
        params,
        run_in_subprocess,
        capture_output,
    )
