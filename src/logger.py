import logging
import os
from datetime import datetime

LOGS_GIR = "logs"
os.makedirs(LOGS_GIR, exist_ok=True)

_run_started_at = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = os.path.join(LOGS_GIR, f"run_{_run_started_at}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

def get_logger(name:str)->logging.Logger:
    return logging.getLogger(name)
    