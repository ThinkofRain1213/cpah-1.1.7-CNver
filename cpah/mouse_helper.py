## The majority of this file is taken from:
## https://stackoverflow.com/a/4263994
import ctypes
import time

from typing import Tuple

import win32api  # type: ignore

from . import constants


class Mouse:
    """Mouse operation wrapper class"""

    MOUSEEVENTF_MOVE = 0x0001  ## Mouse move
    MOUSEEVENTF_LEFTDOWN = 0x0002  ## Left button down
    MOUSEEVENTF_LEFTUP = 0x0004  ## Left button up
    MOUSEEVENTF_ABSOLUTE = 0x8000  ## Absolute move
    SM_CXSCREEN = 0
    SM_CYSCREEN = 1

    def _do_event(
        self,
        flags: int,
        x_position: int = 0,
        y_position: int = 0,
        data: int = 0,
        extra_info: int = 0,
    ):
        """Executes a mouse event."""
        x_size = ctypes.windll.user32.GetSystemMetrics(self.SM_CXSCREEN)
        y_size = ctypes.windll.user32.GetSystemMetrics(self.SM_CYSCREEN)
        x = int(65536 * x_position / x_size + 1)
        y = int(65536 * y_position / y_size + 1)
        return ctypes.windll.user32.mouse_event(flags, x, y, data, extra_info)

    def move(self, position: Tuple[int, int]):
        """
        Helper function for resetting the relative position and
        moving the mouse to the given position.
        """
        ## Yes, this is necessary. Don't ask me why because I don't know.
        ## My best guess is that internally, the game keeps the cursor centered,
        ## thus making all mouse movement relative to where the cursor is in-game.
        self._raw_move((99999, 99999))
        time.sleep(0.001)
        self._raw_move((0, 0))
        time.sleep(0.001)
        self._raw_move(position)
        time.sleep(constants.MOUSE_MOVE_DELAY)

    def _raw_move(self, position: Tuple[int, int]):
        """Moves the mouse to the specified coordinates."""
        (x, y) = position
        old_position = self.get_position()
        x = x if (x != -1) else old_position[0]
        y = y if (y != -1) else old_position[1]
        self._do_event(self.MOUSEEVENTF_MOVE + self.MOUSEEVENTF_ABSOLUTE, x, y)

    def click(self):
        """Left click at the current cursor location."""
        self._do_event(self.MOUSEEVENTF_LEFTDOWN + self.MOUSEEVENTF_LEFTUP)
        time.sleep(constants.MOUSE_CLICK_DELAY)

    def get_position(self) -> Tuple[int, int]:
        """Gets the current mouse position."""
        return win32api.GetCursorPos()


MOUSE = Mouse()
