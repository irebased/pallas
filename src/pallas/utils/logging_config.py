import logging
import os
from pathlib import Path
from typing import Optional

def setup_logger(name: str, log_level: int = logging.INFO, log_to_console: bool = False, run_id: Optional[str] = None) -> logging.Logger:
    """Set up a logger with file and optional console handlers.

    Args:
        name: Name of the logger (typically __name__ of the module)
        log_level: Logging level (default: INFO)
        log_to_console: Whether to also log to console (default: False)
        run_id: Optional UUID to include in the log filename

    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Remove existing handlers to avoid duplicates
    logger.handlers = []

    # Create file handler with optional run_id in filename
    base_filename = name.replace('.', '_')
    filename = f"{base_filename}_{run_id}.log" if run_id else f"{base_filename}.log"
    log_file = log_dir / filename
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)

    # Create console handler if requested
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # Create formatter and add it to the file handler
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)

    return logger

def get_logger(name: str, verbose: bool = False, run_id: Optional[str] = None) -> logging.Logger:
    """Get a configured logger instance.

    Args:
        name: Name of the logger (typically __name__ of the module)
        verbose: Whether to enable verbose console output (default: False)
        run_id: Optional UUID to include in the log filename

    Returns:
        logging.Logger: Configured logger instance
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    return setup_logger(name, log_level, verbose, run_id)