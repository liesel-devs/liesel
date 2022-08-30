import logging
from pathlib import Path


def setup_logger() -> None:
    """
    Sets up a basic `StreamHandler`, which prints log messages to the terminal.
    The default log level of the `StreamHandler` is set to "info".

    The global logging level for liesel log output can be adjusted like this::

        import logging
        logger = logging.getLogger("liesel")
        logger.level = logging.WARNING

    This will set the log level to "warning".
    """

    # We adjust only our library's logger
    logger = logging.getLogger("liesel")

    # This is the level that will in principle be handled by the logger
    # If it is set, for example, to logging.WARNING, this logger will never
    # emit messages of a level below warning.
    logger.setLevel(logging.DEBUG)

    # By setting this to False, we prevent the liesel log messages from being passed
    # on to the root logger. This prevents duplication of the log messages.
    logger.propagate = False

    # This is the default handler that we set for our log messages
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # We define the format of log messages for this handler
    formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)

    logger.addHandler(ch)


def reset_logger() -> None:
    """
    Resets the liesel logger.

    Specifically, this function ...

    - ... resets the level of the ``"liesel"`` logger to ``logging.NOTSET``.
    - ... sets ``propagate=True`` for the ``"liesel"`` logger.
    - ... removes ALL handlers set to the ``"liesel"`` logger.

    This function is useful if you want to set up a custom logging configuration.

    """
    # We adjust only our library's logger
    logger = logging.getLogger("liesel")

    # Removes the level of the logger. Thus, this logger will propagate all log records.
    logger.setLevel(logging.NOTSET)

    # By setting this to True, we allow the liesel log messages to be passed
    # on to the root logger.
    logger.propagate = True

    # Removes all handlers from the liesel logger
    for handler in logger.handlers:
        logger.removeHandler(handler)


def add_file_handler(
    path: str | Path,
    level: str,
    logger: str = "liesel",
    fmt: str = "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
) -> None:
    """
    Adds a file handler for logging output.

    ## Arguments

    - `path`: Absolute path to log file. If it does not exist, it will be created.
      If any parent directory does not exist, it will be created as well.
    - `level`: The level of messages to log to the specified file. Can be "debug",
      "info", "warning", "error" or "critical". The logger will catch all messages
      from the specified level upwards.
    - `logger`: The name of the logger to configure the file handler for. Can be,
      for instance, the full name of a Liesel module. For example, to configure a
      logger for `liesel.goose`, the logger should be specified as "liesel.goose".
    - `fmt`: Formatting string, see the documentation of the standard library's
      `logging.Formatter` class.

    ## Examples

    A basic file handler, catching all log messages in the Liesel framework:

    ```python
    import logging
    import liesel as lsl

    lsl.logging.add_file_handler(path="path/to/logfile.log", level="debug")

    logger = logging.getLogger("liesel")
    logger.warning("My warning message")
    ```

    A file handler that catches only log messages from the `liesel.goose` module of
    level "warning" or higher:

    ```python
    import logging
    import liesel as lsl

    lsl.logging.add_file_handler(
        path="path/to/goose_warnings.log",
        level="warning",
        logger="liesel.goose"
    )

    logger = logging.getLogger("liesel")
    logger.warning("My warning message")
    ```
    """

    path = Path(path)

    if not path.is_absolute():
        raise ValueError("Provided path for logging file handler must be absolute")

    path.parent.mkdir(parents=True, exist_ok=True)

    log = logging.getLogger(logger)
    fh = logging.FileHandler(path)

    fh.setLevel(getattr(logging, level.upper()))

    formatter = logging.Formatter(fmt, "%Y-%m-%d %H:%M:%S")
    fh.setFormatter(formatter)

    log.addHandler(fh)
