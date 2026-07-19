"""
Logging utilities for torchfits.

Library code uses a NullHandler by default so ``import torchfits`` does not
attach a StreamHandler. Applications can add handlers to the ``torchfits``
logger as needed.
"""

import logging

# Re-export stdlib levels so `torchfits.logging.DEBUG` works if the submodule
# shadows the standard `logging` module on the parent package.
DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL

logger = logging.getLogger("torchfits")
logger.addHandler(logging.NullHandler())
