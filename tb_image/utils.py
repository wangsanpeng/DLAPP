#!/usr/bin/env python
# coding=utf-8

import functools
import random
import time
import logging


def type_check(f):
    """
    A wrapper function to check the correctness of the arguments and return values of function 'f'
    :param f:
    :return:
    """
    def do_type_check(name, arg):
        expected_type = f.__annotations__.get(name, None)
        if expected_type and not isinstance(arg, expected_type):
            raise RuntimeError(
                "{} should be of type {} instead of {}".format(name, expected_type.__name__, type(arg).__name__))

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        for i, arg in enumerate(args[:f.__code__.co_nlocals]):
            do_type_check(f.__code__.co_varnames[i], arg)
        for name, arg in kwargs.items():
            do_type_check(name, arg)

        result = f(*args, **kwargs)

        do_type_check('return', result)
        return result

    return wrapper


def retry(initial_delay,
          max_delay,
          factor=2.0,
          jitter=0.25,
          is_retriable=None):
    """Simple decorator for wrapping retriable functions.

  Args:
    initial_delay: the initial delay.
    factor: each subsequent retry, the delay is multiplied by this value.
        (must be >= 1).
    jitter: to avoid lockstep, the returned delay is multiplied by a random
        number between (1-jitter) and (1+jitter). To add a 20% jitter, set
        jitter = 0.2. Must be < 1.
    max_delay: the maximum delay allowed (actual max is
        max_delay * (1 + jitter).
    is_retriable: (optional) a function that takes an Exception as an argument
        and returns true if retry should be applied.
  """
    if factor < 1:
        raise ValueError('factor must be >= 1; was %f' % (factor,))

    if jitter >= 1:
        raise ValueError('jitter must be < 1; was %f' % (jitter,))

    # Generator to compute the individual delays
    def delays():
        delay = initial_delay
        while delay <= max_delay:
            yield delay * random.uniform(1 - jitter, 1 + jitter)
            delay *= factor

    def wrap(fn):
        """Wrapper function factory invoked by decorator magic."""

        def wrapped_fn(*args, **kwargs):
            """The actual wrapper function that applies the retry logic."""
            for delay in delays():
                try:
                    return fn(*args, **kwargs)
                except Exception as e:  # pylint: disable=broad-except)
                    if is_retriable is None:
                        continue

                    if is_retriable(e):
                        time.sleep(delay)
                    else:
                        raise
            return fn(*args, **kwargs)

        return wrapped_fn

    return wrap


def _create_logger(log_level, log_format="", log_file=""):
    if log_format == "":
        log_format = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s'

    logger = logging.getLogger("")
    logger.setLevel(log_level)

    formatter = logging.Formatter(log_format)
    if log_file != "":
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(log_level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger

logger = _create_logger(logging.INFO)
