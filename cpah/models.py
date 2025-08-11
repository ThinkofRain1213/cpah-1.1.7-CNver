import dataclasses
import traceback

from typing import Dict, Iterable, List, Optional, Tuple

import ahk  # type: ignore
import numpy  # type: ignore
import pyautogui  # type: ignore
pyautogui.FAILSAFE = False
import pydantic
import system_hotkey  # type: ignore

from . import constants
from . import exceptions
from .logger import LOG


class BaseAutohackKeyBindings(pydantic.BaseModel):
    activation: str = pydantic.Field("f", description="UI activation (use) key")
    up: str = pydantic.Field("Up", description="UI menu up key")
    down: str = pydantic.Field("Down", description="UI menu down key")
    left: str = pydantic.Field("Left", description="UI menu left key")
    right: str = pydantic.Field("Right", description="UI menu right key")


class AutohackKeyBindings(BaseAutohackKeyBindings):
    activation: str = pydantic.Field("f", description="UI activation (use) key")
    up: str = pydantic.Field("up", description="UI menu up key")
    down: str = pydantic.Field("down", description="UI menu down key")
    left: str = pydantic.Field("left", description="UI menu left key")
    right: str = pydantic.Field("right", description="UI menu right key")

    @pydantic.validator("*")
    def key_validator(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Key must be defined")
        if value not in pyautogui.KEY_NAMES:
            raise ValueError(f"{value} is not a valid key")
        return value


class Config(pydantic.BaseSettings):
    schema_version: int = pydantic.Field(
        constants.CONFIG_SCHEMA_VERSION, description="Config version number"
    )
    show_autohack_warning_message: bool = pydantic.Field(
        True, description="Show the autohack warning message to not move the mouse"
    )
    auto_autohack: bool = pydantic.Field(
        False,
        description="Run autohacking automatically if all daemons can be obtained",
    )
    enable_beeps: bool = pydantic.Field(
        False, description="Allow for beep notifications"
    )
    analysis_hotkey: List[str] = pydantic.Field(
        ["control", "shift", "h"], description="Hotkey for kicking off analysis"
    )
    autohack_hotkey: List[str] = pydantic.Field(
        ["control", "shift", "k"], description="Hotkey for kicking off the autohack"
    )
    game_focus_delay: int = pydantic.Field(
        400,
        description="Amount of milliseconds to wait after the game is focused",
        ge=50,
        le=10000,
    )
    buffer_size_override: int = pydantic.Field(
        0, description="Buffer size manual override", ge=0, le=10
    )
    matrix_code_detection_threshold: float = pydantic.Field(
        0.7,
        description="Detection threshold for codes in the matrix",
        gt=0.0,
        lt=1.0,
    )
    buffer_box_detection_threshold: float = pydantic.Field(
        0.7,
        description="Detection threshold for buffer boxes",
        gt=0.0,
        lt=1.0,
    )
    sequence_code_detection_threshold: float = pydantic.Field(
        0.7,
        description="Detection threshold for codes in the sequences",
        gt=0.0,
        lt=1.0,
    )
    daemon_detection_threshold: float = pydantic.Field(
        0.7,
        description="Detection threshold for sequence daemon names",
        gt=0.0,
        lt=1.0,
    )
    core_detection_threshold: float = pydantic.Field(
        0.7,
        description=(
            "Detection threshold for core elements of the breach protocol screen"
        ),
        gt=0.0,
        lt=1.0,
    )
    detection_language: str = pydantic.Field(
        "English", description="Breach protocol screen language for analysis"
    )
    force_autohack: bool = pydantic.Field(
        False, description="Changes analysis to always present a solvable solution"
    )
    daemon_priorities: List[constants.Daemon] = pydantic.Field(
        list(), description="Daemons to always keep during a forced autohack"
    )
    autohack_key_bindings: AutohackKeyBindings = pydantic.Field(
        AutohackKeyBindings(), description="Key bindings for standard autohacking"
    )
    autohack_keypress_delay: int = pydantic.Field(
        17,
        description="Millisecond delay between keypresses during the autohack",
        ge=0,
        le=500,
    )
    window_title: str = pydantic.Field(
        constants.GAME_EXECUTABLE_TITLE,
        description="Window title to focus",
        min_length=1,
    )
    ahk_enabled: bool = pydantic.Field(
        False, description="Use AutoHotkey for autohacking"
    )
    ahk_executable: str = pydantic.Field(
        "", description="AutoHotkey.exe executable location"
    )
    ahk_autohack_key_bindings: BaseAutohackKeyBindings = pydantic.Field(
        BaseAutohackKeyBindings(),
        description="Key bindings for AutoHotkey assisted autohacking",
    )
    sequential_hotkey_actions: bool = pydantic.Field(
        False, description="Enable sequential hotkey actions"
    )
    sequential_hotkey_actions_timeout: int = pydantic.Field(
        5000,
        description="Millisecond timeout for next hotkey action",
        ge=1000,
        le=30000,
    )
    daemon_toggle_hotkey: List[str] = pydantic.Field(
        ["control", "shift"], description="Hotkey for kicking off analysis"
    )

    @pydantic.validator("analysis_hotkey", "autohack_hotkey")
    def common_hotkey_sequence_validator(
        cls, value: List[str], field: pydantic.fields.ModelField
    ) -> List[str]:
        valid = set(system_hotkey.vk_codes).union(set(system_hotkey.win_modders))
        invalid_keys = set(value).difference(valid)
        if invalid_keys:
            LOG.error(f"{field.name} has invalid keys: {invalid_keys}")
            raise ValueError(
                f"The following keys are invalid: {', '.join(invalid_keys)}"
            )

        if value:
            hotkey = system_hotkey.SystemHotkey()
            try:
                hotkey.parse_hotkeylist(value)
            except Exception as exception:
                LOG.exception(f"{field.name} sequence invalid:")
                nice_name = field.name.replace("_", " ").capitalize()
                raise ValueError(f"{nice_name} sequence invalid: {exception}")
        return value

    @pydantic.validator("daemon_toggle_hotkey")
    def daemon_toggle_hotkey_sequence_validator(cls, value: List[str]) -> List[str]:
        valid = set(system_hotkey.vk_codes).union(set(system_hotkey.win_modders))
        invalid_keys = set(value).difference(valid)
        if invalid_keys:
            raise ValueError(
                f"The following keys are invalid: {', '.join(invalid_keys)}"
            )

        if value:
            hotkey = system_hotkey.SystemHotkey()
            try:
                hotkey.parse_hotkeylist(value + ["1"])
            except Exception as exception:
                LOG.exception("daemon_toggle_hotkey sequence invalid:")
                raise ValueError(f"Daemon toggle hotkey sequence invalid: {exception}")
        return value

    @pydantic.validator("detection_language")
    def detection_language_validator(cls, value: str) -> str:
        if value not in constants.TEMPLATE_LANGUAGE_DATA:
            raise ValueError(f"Detection language {value} is not supported")
        return value


@dataclasses.dataclass
class ScreenshotData:
    screenshot: numpy.ndarray


@dataclasses.dataclass(eq=True, frozen=True)
class BreachProtocolData:
    data: Tuple[Tuple[int, ...], ...]
    matrix_size: int
    buffer_size: int
    sequences: Tuple[Tuple[int, ...], ...]
    daemons: Tuple[Optional[constants.Daemon], ...]
    daemon_names: Tuple[str, ...]


@dataclasses.dataclass
class ScreenBounds:
    code_matrix: Tuple[Tuple[int, int], Tuple[int, int]]
    buffer_box: Tuple[Tuple[int, int], Tuple[int, int]]
    sequences: Tuple[Tuple[int, int], Tuple[int, int]]
    daemons: Tuple[Tuple[int, int], Tuple[int, int]]


@dataclasses.dataclass
class AnalysisData:
    breach_protocol_data: BreachProtocolData
    screenshot_data: ScreenshotData
    screen_bounds: ScreenBounds


@dataclasses.dataclass
class SequencePathData:
    all_sequence_paths: Tuple[Tuple[int, ...], ...]
    shortest_solution: Optional[Tuple[int, ...]]
    shortest_solution_path: Optional[Tuple[Tuple[int, int], ...]]
    solution_valid: bool
    computationally_complex: bool


@dataclasses.dataclass(eq=True, frozen=True)
class Sequence:
    string: str
    contiguous_block_indices: Tuple[int, ...]


@dataclasses.dataclass
class ConvertedSequence:
    data: Tuple[int, ...]
    contiguous_block_indices: Tuple[int, ...]


@dataclasses.dataclass
class SequenceSelectionData:
    daemon: Optional[constants.Daemon]
    daemon_name: str
    sequence: Tuple[int, ...]
    selected: bool


class Error:
    """Exception container with additional information"""

    def __init__(self, exception: Exception):
        message = str(exception)
        traceback_string = "\n".join(traceback.format_tb(exception.__traceback__))
        if isinstance(exception, exceptions.CPAHException):
            critical = exception.critical
            unhandled = False
        else:
            unhandled = critical = True

        if unhandled:
            message = (
                f"An unhandled error occurred ({exception.__class__.__name__}):"
                f"\n{message}\n\n{traceback_string}"
            )

        self.exception = exception
        self.traceback = traceback_string
        self.unhandled = unhandled
        self.critical = critical
        self.message = message
        self.title = f"Error (CPAH {constants.VERSION})"

    def __str__(self):
        return (
            "<Error "
            f"exception=<{self.exception}>, "
            f'traceback="{self.traceback}", '
            f"unhandled={self.unhandled}, "
            f"critical={self.critical}, "
            f'message="{self.message}">'
        )
