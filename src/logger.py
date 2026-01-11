"""
Structured logging setup for HAR PhD Thesis project.

Provides centralized logging configuration with file and console handlers.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    log_dir: Optional[Path] = None,
    log_level: str = "INFO",
    console_level: str = "INFO",
    file_level: str = "DEBUG"
) -> logging.Logger:
    """
    Setup structured logger with file and console handlers.
    
    Args:
        name: Logger name (typically __name__ or script name)
        log_dir: Directory for log files (default: logs/)
        log_level: Overall log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console_level: Console handler log level
        file_level: File handler log level
    
    Returns:
        Configured logger instance
    """
    if log_dir is None:
        log_dir = Path("logs")
    
    # Create log directory if it doesn't exist
    log_dir.mkdir(exist_ok=True)
    
    # Get or create logger
    logger = logging.getLogger(name)
    
    # Avoid adding handlers multiple times (if logger already configured)
    if logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # File handler - detailed logs with date in filename
    log_file = log_dir / f"{name}_{datetime.now():%Y%m%d}.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(getattr(logging, file_level.upper(), logging.DEBUG))
    
    # Console handler - less verbose for readability
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, console_level.upper(), logging.INFO))
    
    # Formatter - structured format
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    file_handler.setFormatter(detailed_formatter)
    console_handler.setFormatter(simple_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get or create a logger instance.
    
    If logger doesn't exist, creates one with default settings.
    If logger exists, returns existing instance.
    
    Args:
        name: Logger name (default: root logger)
    
    Returns:
        Logger instance
    """
    if name is None:
        name = "har_phd_thesis"
    
    logger = logging.getLogger(name)
    
    # If logger has no handlers, set it up with defaults
    if not logger.handlers:
        return setup_logger(name)
    
    return logger
