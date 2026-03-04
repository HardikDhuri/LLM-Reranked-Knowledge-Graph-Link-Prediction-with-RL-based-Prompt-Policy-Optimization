"""Tests for src/utils/logging_config.py."""

import logging

from src.utils.logging_config import setup_logging


def test_setup_logging_returns_logger(monkeypatch, tmp_path):
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "test-key")
    logger = setup_logging(
        level="INFO",
        log_dir=str(tmp_path),
        log_to_file=False,
        log_to_console=True,
    )
    assert isinstance(logger, logging.Logger)


def test_setup_logging_correct_level(monkeypatch, tmp_path):
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "test-key")
    logger = setup_logging(
        level="WARNING",
        log_dir=str(tmp_path),
        log_to_file=False,
        log_to_console=True,
    )
    assert logger.level == logging.WARNING


def test_console_handler_added(monkeypatch, tmp_path):
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "test-key")
    logger = setup_logging(
        level="INFO",
        log_dir=str(tmp_path),
        log_to_file=False,
        log_to_console=True,
    )
    stream_handlers = [
        h for h in logger.handlers if isinstance(h, logging.StreamHandler)
    ]
    assert len(stream_handlers) >= 1


def test_no_console_handler_when_disabled(monkeypatch, tmp_path):
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "test-key")
    logger = setup_logging(
        level="INFO",
        log_dir=str(tmp_path),
        log_to_file=False,
        log_to_console=False,
    )
    # No StreamHandler (which is the base class for FileHandler too, but we check
    # that there are no handlers at all when both are disabled)
    assert len(logger.handlers) == 0


def test_file_handler_creates_log_file(monkeypatch, tmp_path):
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "test-key")
    setup_logging(
        level="INFO",
        log_dir=str(tmp_path),
        log_to_file=True,
        log_to_console=False,
    )
    log_files = list((tmp_path / "logs").glob("run_*.log"))
    assert len(log_files) == 1


def test_file_handler_with_experiment_name(monkeypatch, tmp_path):
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "test-key")
    setup_logging(
        level="INFO",
        log_dir=str(tmp_path),
        log_to_file=True,
        log_to_console=False,
        experiment_name="myexp",
    )
    log_files = list((tmp_path / "logs").glob("run_myexp_*.log"))
    assert len(log_files) == 1


def test_noisy_loggers_suppressed(monkeypatch, tmp_path):
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "test-key")
    setup_logging(
        level="DEBUG",
        log_dir=str(tmp_path),
        log_to_file=False,
        log_to_console=False,
    )
    assert logging.getLogger("urllib3").level == logging.WARNING
    assert logging.getLogger("requests").level == logging.WARNING
    assert logging.getLogger("pykeen").level == logging.WARNING
