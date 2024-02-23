"""Logging utils."""
import logging
import os
from typing import ClassVar


class Logger:
    """Logger."""

    COLOR_CODES: ClassVar[dict[str, str]] = {
        "DEBUG": "\033[96m",       # Light cyan
        "INFO": "\033[94m",        # Light blue
        "WARNING": "\033[93m",     # Light yellow
        "ERROR": "\033[91m",       # Light red
        "CRITICAL": "\033[41m",    # Red background
        "ENDC": "\033[0m",
    } if os.environ.get("ENV") == "dev" else {}

    @classmethod
    def log(
        cls: type["Logger"],
        msg: str,
        color: str | None  = None,
        *args: str,
    ) -> None:
        """log."""
        color_code = cls.COLOR_CODES.get(color, "")
        endc = cls.COLOR_CODES.get("ENDC", "")
        print(f"{color_code}{str(msg) % args}{endc}")  # noqa: T201

    @classmethod
    def debug(cls: type["Logger"], msg: str, *args: str) -> None:
        """debug."""
        cls.log(msg, "DEBUG", *args)

    @classmethod
    def info(cls: type["Logger"], msg: str, *args: str) -> None:
        """info."""
        cls.log(msg, "INFO", *args)

    @classmethod
    def warning(cls: type["Logger"], msg: str, *args: str) -> None:
        """warning."""
        cls.log(msg, "WARNING", *args)

    @classmethod
    def error(cls: type["Logger"], msg: str, *args: str) -> None:
        """error."""
        cls.log(msg, "ERROR", *args)

    @classmethod
    def critical(cls: type["Logger"], msg: str, *args: str) -> None:
        """critical."""
        cls.log(msg, "CRITICAL", *args)


def get_logger(
    in_prefect: bool = True, name: str | None = None,
) -> logging.Logger | Logger:
    """get_logger."""
    if in_prefect:
        return Logger

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)8s | <%(name)s> %(module)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    return logging.getLogger(name=name)


if __name__ == "__main__":
    # Usage
    logger = get_logger()
    logger.debug("Debugging %s", "details")
    logger.info("Information about %s", "something")
    logger.warning("A warning related to %s", "configuration")
    logger.error("An error occurred during %s", "execution")
    logger.critical("Critical issue with %s", "system")
