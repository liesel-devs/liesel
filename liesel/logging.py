"""
Logging utilities.
"""

import logging
from pathlib import Path


def setup_logger() -> None:
    """
    Sets up a basic ``StreamHandler`` that prints log messages to the terminal.
    The default log level of the ``StreamHandler`` is set to "info".

    The global log level for Liesel can be adjusted like this::

        import logging
        logger = logging.getLogger("liesel")
        logger.level = logging.WARNING

    This will set the log level to "warning".
    """

    # We adjust only our library's logger
    logger = logging.getLogger("liesel")

    # This is the level that will in principle be handled by the logger.
    # If it is set, for example, to logging.WARNING, this logger will never
    # emit messages of a level below warning
    logger.setLevel(logging.INFO)

    # By setting this to False, we prevent the Liesel log messages from being passed on
    # to the root logger. This prevents duplication of the log messages
    logger.propagate = False

    # This is the default handler that we set for our log messages
    handler = logging.StreamHandler()

    # We define the format of log messages for this handler
    formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)

    logger.addHandler(handler)


def reset_logger() -> None:
    """
    Resets the Liesel logger.

    Specifically, this function...

    - ... resets the level of the Liesel logger to ``logging.NOTSET``.
    - ... sets ``propagate=True`` for the Liesel logger.
    - ... removes *all* handlers from the Liesel logger.

    This function is useful if you want to set up a custom logging configuration.
    """

    # We adjust only our library's logger
    logger = logging.getLogger("liesel")

    # Removes the level of the logger. All log messages will be propagated
    logger.setLevel(logging.NOTSET)

    # By setting this to True, we allow the Liesel log messages to be passed on
    # to the root logger
    logger.propagate = True

    # Removes all handlers from the Liesel logger
    for handler in logger.handlers:
        logger.removeHandler(handler)


def add_file_handler(
    path: str | Path,
    level: str,
    logger: str = "liesel",
    fmt: str = "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
) -> None:
    """
    Adds a file handler to a logger.

    Parameters
    ----------
    path
        Absolute path to the log file. If it does not exist, it will be created.
        If any parent directory does not exist, it will be created as well.
    level
        The log level of the messages to write to the file. Can be ``"debug"``,
        ``"info"``, ``"warning"``, ``"error"`` or ``"critical"``. The file will
        contain all messages from the specified level upwards.
    logger
        The name of the logger to configure the file handler for. Can be, for example,
        the full name of a Liesel module. For :mod:`liesel.goose`, the argument should
        be specified as ``"liesel.goose"``, etc.
    fmt
        Formatting string. See the documentation of the :class:`logging.Formatter`.

    Examples
    --------
    A basic file handler catching all log messages from Liesel::

        import logging
        import liesel as lsl

        lsl.logging.add_file_handler(path="path/to/logfile.log", level="debug")

        logger = logging.getLogger("liesel")
        logger.warning("My warning message")

    A file handler that catches only log messages from the :mod:`liesel.goose` module
    of level "warning" or higher::

        import logging
        import liesel as lsl

        lsl.logging.add_file_handler(
            path="path/to/goose_warnings.log",
            level="warning",
            logger="liesel.goose"
        )

        logger = logging.getLogger("liesel")
        logger.warning("My warning message")
    """

    path = Path(path)

    if not path.is_absolute():
        raise ValueError("Provided path for logging file handler must be absolute")

    path.parent.mkdir(parents=True, exist_ok=True)

    _logger = logging.getLogger(logger)
    handler = logging.FileHandler(path)

    handler.setLevel(getattr(logging, level.upper()))

    formatter = logging.Formatter(fmt, "%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)

    _logger.addHandler(handler)
