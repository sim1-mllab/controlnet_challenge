import logging
from functools import wraps
from typing import Callable

# Option 1
_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(funcName)s() --  %(message)s"

_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT, datefmt=_DATE_FORMAT)


def df_info_to_dict(df):
    """Return dictionary containing df_info() style information

    :param df: DataFrame to info'
    :type df: DataFrame
    """
    return dict(
        dtypes=df.dtypes.to_dict(), shape=df.shape, index_type=type(df.index).__name__
    )


def get_logger(name: str) -> logging.Logger:
    """Get logger

    :param name: name of the logger
    :type name: str
    :return: logger instance
    :rtype: logging.Logger
    """
    if "src/" in name:
        name = name.split("src/")[-1]

    return logging.getLogger(name)


def log_start_finish(logger: logging.Logger) -> Callable:
    """Decorator for logging start and finish of a function.

    :param logger: Logger to be used in decorated function
    :return: wrapper function
    """

    @wraps(logger)
    def decorate(func):
        def call(*args, **kwargs):
            # Call function with logger at the start and after finish
            logger.info(f">> Started '{func.__name__}()'")
            result = func(*args, **kwargs)
            logger.info(f"<< Finished '{func.__name__}()'")

            return result

        return call

    return decorate
