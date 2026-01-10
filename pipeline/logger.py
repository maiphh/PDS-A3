"""
Logging utility for the pipeline project.
"""

import logging
import os
import sys
from datetime import datetime

def setup_logger(name: str = "pipeline", log_dir: str = "pipeline/output/logs", level=logging.INFO):
    """
    Set up a logger that writes to both console and a log file.
    
    Parameters:
    -----------
    name : str
        Name of the logger.
    log_dir : str
        Directory to save log files.
    level : int
        Logging level (default: logging.INFO)
        
    Returns:
    --------
    logging.Logger
        Configured logger instance.
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Check if handlers are already configured to avoid duplicates
    if logger.handlers:
        return logger
        
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    print(f"Logging configured. Log file: {log_file}")
    
    return logger
