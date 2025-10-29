"""
Safer version of the grader:
 * Disable forking
   RLIMIT_NPROC = 1
   Disable pytorch threading first, since it forks
 * Disable print statements
   Override stdout and stderr
 * Returns value 10-110 for successful runs
 * Wrap this grader call in `unshare -rn ...` to disable networking
 * `exit` command disabled
"""

import builtins
import signal
import sys
import time
import traceback

from .grader import (
    CheckFailed,
    grade_all,
    init_loggers,
    list_all_kwargs,
    load_assignment,
)

alarm = signal.alarm


def disable_alarm(disable=True):
    def _alarm(*args, **kwargs):
        # This will help us flag down whoever is trying to break stuff
        raise Exception("Function disabled. Please contact staff.")

    if disable:
        signal.alarm = _alarm


def case(func, kwargs=None, score=1, extra_credit=False, timeout=1):
    """
    Use @Case(score, extra_credit) as a decorator for member functions of a Grader
    A test case can return a value between 0 and 1 as a score
    If the test fails it should raise an assertion
    The test case may optionally return a tuple (score, message)
    """
    if kwargs is None:
        kwargs = {}

    def handle_timeout(signum, frame):
        raise TimeoutError(f"Timeout after {timeout} sec")

    def wrapper(self):
        n_passed = 0.0
        total = 0.0
        msg = "passed"
        error = ""

        # yields multiple for multi-case
        for a in list_all_kwargs(**kwargs):
            tick = time.time()

            signal.signal(signal.SIGALRM, handle_timeout)
            alarm(timeout)

            try:
                v = func(self, **a)
                alarm(0)

                if v is None:
                    v = 1
                elif isinstance(v, tuple):
                    v, msg = v
                else:
                    assert isinstance(
                        v, float
                    ), f"case returned {v!r} which is not a float!"

                n_passed += v
            except TimeoutError as e:
                msg = str(e)
            except NotImplementedError:
                msg = "Not Implemented"
            except CheckFailed as e:
                msg = str(e)
            except Exception as e:
                msg = type(e).__name__
                error = traceback.format_exc()

            tock = time.time()

            if tock - tick > timeout + 0.1:
                n_passed = 0.0
                break

            total += 1

        return int(n_passed * score / total + 0.5), msg, error

    wrapper.score = score
    wrapper.extra_credit = extra_credit
    wrapper.__doc__ = func.__doc__
    wrapper.func = func

    return wrapper


class Case:
    def __init__(self, score=1, extra_credit=False, timeout=1):
        self.score = score
        self.extra_credit = extra_credit
        self.timeout = timeout

    def __call__(self, func):
        return case(
            func, score=self.score, extra_credit=self.extra_credit, timeout=self.timeout
        )


class MultiCase:
    def __init__(self, score=1, extra_credit=False, **kwargs):
        self.score = score
        self.extra_credit = extra_credit
        self.kwargs = kwargs

    def __call__(self, func):
        return case(
            func, kwargs=self.kwargs, score=self.score, extra_credit=self.extra_credit
        )


def disable_exit(disable=True):
    e = sys.exit
    if disable:
        sys.exit = lambda x: None
    del builtins.exit
    return e


def disable_file_write(disable=True):
    import resource

    if disable:
        resource.setrlimit(resource.RLIMIT_FSIZE, (10 * 1024 * 1024, 10 * 1024 * 1024))


def run():
    import argparse

    parser = argparse.ArgumentParser("Grade your assignment")
    parser.add_argument("assignment", default="homework")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-vv", "--very_verbose", action="store_true")
    parser.add_argument("-n", "--nolimit", dest="limit", action="store_false")
    parser.add_argument("--log_path", default=None)

    args = parser.parse_args()

    logger = init_loggers(args.log_path, show_debug=args.very_verbose)
    e = disable_exit(args.limit)
    disable_alarm(args.limit)

    logger.info("Loading assignment")
    assignment = load_assignment(
        logger, args.assignment, pre_import_fn=lambda: disable_file_write(args.limit)
    )

    if assignment is None:
        e(0)

    logger.info("Grading")
    score = grade_all(assignment, logger, args.verbose or args.very_verbose)

    # offset score so it doesn't overlap with standard exit codes
    e(score + 10)
