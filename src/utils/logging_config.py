"""Structured logging configuration for the project."""

import logging
import sys
from datetime import datetime
from pathlib import Path

from src.config import get_settings


def setup_logging(
    level: str = None,
    log_dir: str = None,
    log_to_file: bool = True,
    log_to_console: bool = True,
    experiment_name: str = None,
) -> logging.Logger:
    """
    Configure structured logging for the project.
    - Console: concise format
    - File: detailed format with timestamps
    """
    settings = get_settings()
    level = level or settings.log_level
    log_dir = Path(log_dir or settings.results_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
        console_fmt = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
        console_handler.setFormatter(console_fmt)
        root_logger.addHandler(console_handler)

    # File handler
    if log_to_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name_part = f"_{experiment_name}" if experiment_name else ""
        log_file = log_dir / f"run{name_part}_{timestamp}.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)  # Always capture everything to file
        file_fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d"
            " | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_fmt)
        root_logger.addHandler(file_handler)

    # Suppress noisy third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("pykeen").setLevel(logging.WARNING)

    file_status = "enabled" if log_to_file else "disabled"
    root_logger.info(f"Logging initialized (level={level}, file={file_status})")
    return root_logger
