from typing import Iterable, Optional

import system_hotkey  # type: ignore

from . import constants


class CPAHException(Exception):
    critical = False


class CPAHThreadRunningException(CPAHException):
    pass


class CPAHInvalidConfigException(CPAHException):
    critical = True

    def __init__(self, message: str):
        super().__init__(
            f"The configuration file at {constants.CONFIG_FILE_PATH} "
            f"is invalid or corrupt:\n\n{message}"
        )


class CPAHInvalidNewConfigException(CPAHException):
    def __init__(self, exception: Exception):
        super().__init__(f"The configuration is not valid:\n\n{exception}")


class CPAHHotkeyRegistrationException(CPAHException):
    def __init__(
        self,
        exception: system_hotkey.SystemRegisterError,
        full_bind: Iterable[str] = list(),
    ):
        message = (
            "A hotkey could not be registered. Are there overlapping keybinds, "
            "and/or is another instance of the tool open?\n\n"
            "Some hotkeys will not work until this is resolved.\n\n"
        )
        error_message = str(exception.args[0])
        in_use = error_message.startswith("The bind could be in use elsewhere")
        if not in_use and error_message.startswith("existing bind detected"):
            in_use = True
            full_bind = exception.args[1:]
        if full_bind and in_use:
            message += f"Hotkey already in use: {' + '.join(full_bind)}"
        else:
            message += f"Given error: {exception}"
        super().__init__(message)


class CPAHGameNotFoundException(CPAHException):
    def __init__(self, window_title: str):
        super().__init__(f"Game window not found: {window_title}")


class CPAHScreenshotParseFailedException(CPAHException):
    def __init__(self, message: str):
        super().__init__(
            f"{message}\n\nIs the breach protocol minigame screen active? "
            "If the confidence value is close to the threshold, you can lower the "
            "threshold of core elements detection in the configuration screen.\n\n"
            "Additionally, ensure you have the correct detection language set "
            "in the configuration menu."
        )


class CPAHDataParseFailedException(CPAHException):
    def __init__(
        self,
        message: str,
        detection_type: str = "some",
        post: Optional[str] = None,
    ):
        combined_message = (
            f"{message}\n\nIf you are playing at a resolution smaller than "
            f"{constants.ANALYSIS_IMAGE_SIZE[0]}x{constants.ANALYSIS_IMAGE_SIZE[1]}, "
            f"you may need to decrease {detection_type} detection thresholds. "
            "Additionally, ensure your mouse cursor is not in the way of elements."
        )
        if post:
            combined_message += f"\n\n{post}"
        super().__init__(combined_message)


class CPAHBufferParseFailedException(CPAHDataParseFailedException):
    def __init__(self, message: str):
        super().__init__(
            message,
            detection_type="buffer box",
            post=(
                "If adjusting the detection thresholds does not fix the problem, "
                "you can override the buffer size to bypass automatic detection."
            ),
        )


class CPAHMatrixParseFailedException(CPAHDataParseFailedException):
    def __init__(self, message: str):
        super().__init__(message, detection_type="matrix code")


class CPAHSequenceParseFailedException(CPAHDataParseFailedException):
    def __init__(self, message: str):
        super().__init__(message, detection_type="sequence code")


class CPAHDaemonParseFailedException(CPAHDataParseFailedException):
    def __init__(self, message: str):
        super().__init__(message, detection_type="daemon")


class CPAHAHKException(CPAHException):
    pass


class CPAHAHKNotFoundException(CPAHAHKException):
    def __init__(self):
        super().__init__(
            "AutoHotkey support is enabled, but AutoHotkey.exe was not found"
        )


class CPAHAHKInternalException(CPAHAHKException):
    def __init__(self, message: str):
        super().__init__(
            "The AutoHotkey executable encountered an error. "
            "Check to make sure the AutoHotkey.exe executable path is correct:"
            f"\n\n{message}"
        )
