import logging
import os


def get_logger(name):

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    if os.getenv("VERCEL") != "1":

        os.makedirs("logs", exist_ok=True)

        file_handler = logging.FileHandler("logs/app.log", encoding="utf-8")

        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

    logger.propagate = False

    return logger
