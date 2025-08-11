import logging
import logging.config

from . import constants


logging.config.dictConfig(
    {
        "version": 1,
        "formatters": {
            "default": {"format": "%(asctime)s [%(levelname)s] [CPAH]: %(message)s"}
        },
        "handlers": {
            "default": {
                "level": "DEBUG",
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
            "file": {
                "level": "DEBUG",
                "formatter": "default",
                "class": "logging.handlers.RotatingFileHandler",
                "filename": str(constants.LOG_FILE_PATH),
                "encoding": "utf-8",
                "maxBytes": 500_000,
                "backupCount": 1,
            },
        },
        "loggers": {
            "cpah": {
                "handlers": ["default", "file"],
                "level": "DEBUG",
                "propagate": False,
            },
        },
    }
)
LOG = logging.getLogger("cpah")
