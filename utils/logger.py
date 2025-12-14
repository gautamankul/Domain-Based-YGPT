import logging

LOGGER_NAME = "yil_gpt"

def get_logger(name: str = LOGGER_NAME) -> logging.Logger:
    """Configures and returns a consistent logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger