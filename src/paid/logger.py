"""
This file exports `logger` object for unified logging across the Paid SDK.
"""

import logging
import os

import dotenv

# Configure logging
_ = dotenv.load_dotenv()
logger = logging.getLogger(__name__)

# Set default log level to ERROR, allow override via PAID_LOG_LEVEL environment variable
log_level_name = os.environ.get("PAID_LOG_LEVEL")
if log_level_name is not None:
    log_level = getattr(logging, log_level_name.upper(), logging.ERROR)
else:
    log_level = logging.ERROR

logger.setLevel(log_level)
