import datetime
import logging
import sys
import time

from mokapot import __version__


def make_timer():
    t0 = time.time()

    def elapsed():
        nonlocal t0
        t1 = time.time()
        dt, t0 = t1 - t0, t1
        return dt

    return elapsed


def setup_logging(config):
    # Setup logging
    verbosity_dict = {
        0: logging.ERROR,
        1: logging.WARNING,
        2: logging.INFO,
        3: logging.DEBUG,
    }

    if not config.log_time:
        log_format = "[{levelname}] {message}"
    elif config.max_workers <= 1:
        log_format = "[{asctime}/{levelname}] {message}"
    else:
        log_format = "[{threadName}/{asctime}/{levelname}] {message}"

    logging.basicConfig(
        format=log_format,
        style="{",
        level=verbosity_dict[config.verbosity],
    )

    logging.captureWarnings(True)

    # Stop numba from flooding the log
    numba_logger = logging.getLogger("numba")
    numba_logger.setLevel(logging.WARNING)


def output_start_message(prog_name, config):
    timer = make_timer()
    logging.info(f"{prog_name} version {__version__}")
    logging.info("Written by William E. Fondrie (wfondrie@uw.edu) in the")
    logging.info("Department of Genome Sciences at the University of Washington.")
    logging.info("Command issued:")
    logging.info("  %s", " ".join(sys.argv))
    logging.info("")
    logging.info("Starting Analysis")
    logging.info("=================")
    return timer


def output_end_message(prog_name, config, timer):
    total_time = str(datetime.timedelta(seconds=timer()))

    logging.info("")
    logging.info("=== DONE! ===")
    logging.info(f"{prog_name} analysis completed in {total_time}")
