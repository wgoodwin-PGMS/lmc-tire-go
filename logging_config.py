import logging


def setup_logging(level="INFO"):
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(levelname)8s - %(name)21s - %(message)s",
    )
    logger = logging.getLogger()
    logger.setLevel(numeric_level)
