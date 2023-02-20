from datetime import datetime
import logging # set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt='%Y-%m-%d %H:%M:%S'
)

LOGH_CLI = logging.StreamHandler()
LOGH_FILE = logging.FileHandler("debug.log")
for h in [LOGH_CLI, LOGH_FILE]:
    h.setFormatter(logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s", 
        datefmt='%Y-%m-%d %H:%M:%S'))

import warnings # shut up warnings
warnings.simplefilter("ignore", category=UserWarning)
