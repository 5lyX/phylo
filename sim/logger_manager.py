import logging
import inspect
import os
from enum import Enum, auto


# Define an enum to represent the types of loggers
class LoggerType(Enum):
    DEFAULT = auto()
    FILE_LOGGER = auto()
    ERROR_LOGGER = auto()
    # More logger types can be added as needed


# Create a safe formatter to ensure no exceptions are raised even if attributes are missing
class SafeFormatter(logging.Formatter):
    def format(self, record):
        if not hasattr(record, "class_name"):
            record.class_name = ""
        return super().format(record)


class LoggerManager:
    _loggers = {}

    def __init__(self):
        self._initialize_loggers()

    def _initialize_loggers(self):
        # Define all logger types and their configurations here

        # Default logger: output to the console with the default format
        default_logger = logging.getLogger("default")
        default_logger.setLevel(logging.DEBUG)
        if not default_logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.DEBUG)
            formatter = SafeFormatter(
                "%(asctime)s - %(levelname)s - %(filename)s - %(class_name)s - %(funcName)s - %(message)s"
            )
            handler.setFormatter(formatter)
            default_logger.addHandler(handler)
        self._loggers[LoggerType.DEFAULT] = default_logger

        # File logger: output to a file with a detailed format
        file_logger = logging.getLogger("file_logger")
        file_logger.setLevel(logging.INFO)
        if not file_logger.handlers:
            handler = logging.FileHandler("app.log")
            handler.setLevel(logging.INFO)
            formatter = SafeFormatter(
                "%(asctime)s - %(levelname)s - %(filename)s - %(class_name)s - %(funcName)s - %(message)s"
            )
            handler.setFormatter(formatter)
            file_logger.addHandler(handler)
        self._loggers[LoggerType.FILE_LOGGER] = file_logger

        # Error logger: logs only error-level messages
        error_logger = logging.getLogger("error_logger")
        error_logger.setLevel(logging.ERROR)
        if not error_logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.ERROR)
            formatter = SafeFormatter(
                "%(asctime)s - %(levelname)s - %(filename)s - %(class_name)s - %(funcName)s - %(message)s"
            )
            handler.setFormatter(formatter)
            error_logger.addHandler(handler)
        self._loggers[LoggerType.ERROR_LOGGER] = error_logger

        # More logger types can be added here

    def get_logger(self, logger_type: LoggerType):
        return self._loggers.get(logger_type, self._loggers[LoggerType.DEFAULT])

    def log(self, logger_type: LoggerType, level, message, *args, **kwargs):
        logger = self.get_logger(logger_type)

        # Get the caller's information
        frame = (
            inspect.currentframe().f_back.f_back
        )  # Go back two frames to find the actual caller
        class_name = ""
        if frame is not None and "self" in frame.f_locals:
            class_name = frame.f_locals["self"].__class__.__name__

        # Pass 'class_name' via the 'extra' parameter
        extra = kwargs.get("extra", {})
        extra["class_name"] = class_name
        kwargs["extra"] = extra

        # Use stacklevel=3 to get the correct caller information
        logger.log(level, message, *args, stacklevel=3, **kwargs)

    # Provide shortcut methods so the user doesn't need to specify the logger type
    def debug(self, message, *args, **kwargs):
        self.log(LoggerType.DEFAULT, logging.DEBUG, message, *args, **kwargs)

    def info(self, message, *args, **kwargs):
        self.log(LoggerType.DEFAULT, logging.INFO, message, *args, **kwargs)

    def warning(self, message, *args, **kwargs):
        self.log(LoggerType.DEFAULT, logging.WARNING, message, *args, **kwargs)

    def error(self, message, *args, **kwargs):
        self.log(LoggerType.ERROR_LOGGER, logging.ERROR, message, *args, **kwargs)

    def critical(self, message, *args, **kwargs):
        self.log(LoggerType.ERROR_LOGGER, logging.CRITICAL, message, *args, **kwargs)

    # If specific loggers are needed, more methods can be added
    def file_info(self, message, *args, **kwargs):
        self.log(LoggerType.FILE_LOGGER, logging.INFO, message, *args, **kwargs)

    def file_debug(self, message, *args, **kwargs):
        self.log(LoggerType.FILE_LOGGER, logging.DEBUG, message, *args, **kwargs)
