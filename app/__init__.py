import logging
import sys
from functools import partial

logger = logging.getLogger("app")
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    "%(asctime)s [%(processName)s: %(process)d] [%(threadName)s: %(thread)d] "
    "[%(levelname)s] %(name)s: %(message)s"
)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

logger.error = partial(logger.error, exc_info=True)
logger.exception = partial(logger.exception, exc_info=True)
