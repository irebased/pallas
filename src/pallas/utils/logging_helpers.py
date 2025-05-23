import logging
from typing import Optional
from pallas.utils.logging_config import get_logger

class LoggingHelper:
    """Helper class for logging functionality."""

    def __init__(self, name: str, verbose: bool = False, run_id: Optional[str] = None):
        """Initialize the logging helper.

        Args:
            name: Name of the logger (typically __name__ of the module)
            verbose: Whether to enable verbose logging
            run_id: Optional UUID to include in the log filename
        """
        self.logger = get_logger(name, verbose, run_id)

    def log(self, message: str, level: str = 'info') -> None:
        """Log a message with the specified level.

        Args:
            message: The message to log
            level: The logging level ('debug', 'info', 'warning', 'error', 'critical')
        """
        log_func = getattr(self.logger, level.lower())
        log_func(message)

    def log_error(self, message: str, error: Optional[Exception] = None) -> None:
        """Log an error message with optional exception details.

        Args:
            message: The error message to log
            error: Optional exception to include in the log
        """
        if error:
            self.log(f"{message}: {str(error)}", 'error')
        else:
            self.log(message, 'error')