"""
Logging setup for Chloe AI system using loguru
"""

from loguru import logger


def setup_logger(name: str, log_file: str = None, level: str = "INFO"):
    """Setup logger with console and file handlers using loguru"""
    # Just return the global logger instance
    # Individual files can customize logging as needed
    return logger