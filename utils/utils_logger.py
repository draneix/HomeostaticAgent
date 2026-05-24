import logging

def create_logger(name, log_file, level=logging.INFO):
    """Creates a logger with the specified name and log file."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create console handlers
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)

    # 4. Create the file handler
    fh = logging.FileHandler(log_file, mode='a')
    fh.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger
