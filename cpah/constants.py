## Workaround for fetching monitor sizes that doesn't get affected by window scaling.
## This must be run before pyautogui is imported, as it does some
## behind-the-scenes stuff involving making the process DPI aware,
## which breaks many things involving the screenshot process.
from PySide2.QtWidgets import QApplication

QAPPLICATION_INSTANCE = QApplication([])


import enum
import json
import os
import pathlib
import sys

from typing import Any, Dict, Optional, Tuple

import cv2  # type: ignore
import pyautogui  # type: ignore
pyautogui.FAILSAFE = False
import numpy  # type: ignore

from PIL import Image, ImageDraw, ImageFont  # type: ignore

from PySide2.QtCore import QSettings


## Helper function for reading opencv templates as files
def _rt(template_image_path: pathlib.Path):
    ## SO: 43185605
    with template_image_path.open("rb") as template_file:
        np_array = numpy.asarray(bytearray(template_file.read()), dtype=numpy.uint8)
        return cv2.imdecode(np_array, cv2.IMREAD_COLOR)


DOCUMENTATION_LINK = "https://dreded.gitlab.io/cpah/"

## Detect if running frozen
if hasattr(sys, "_MEIPASS"):
    MODULE_DIRECTORY = pathlib.Path(sys._MEIPASS).resolve()  # type: ignore
else:
    MODULE_DIRECTORY = pathlib.Path(os.path.realpath(__file__)).parent.resolve()
_version_file = MODULE_DIRECTORY / "VERSION"

## General application constants
APPLICATION_NAME = "CP 2077 Auto Hacker"
CONFIG_SCHEMA_VERSION = 6
VERSION = (
    _version_file.read_text().strip() if _version_file.is_file() else "development"
)
MAX_SOLUTION_PATH_LENGTH = 12
MAX_SIZE_ESTIMATE_THRESHOLD = 3500
MEMOIZE_SIZE = 100

## Default window title used to find the process to screenshot
GAME_EXECUTABLE_TITLE = "Cyberpunk 2077 (C) 2020 by CD Projekt RED"

## Directory constants
_qsettings = QSettings(
    QSettings.IniFormat, QSettings.Scope.UserScope, APPLICATION_NAME, application="_"
)
APPLICATION_DIRECTORY = pathlib.Path(_qsettings.fileName()).parent.resolve()

## Make application directory if it doesn't already exist
## Bit of a sketchy place to put this, but oh well
APPLICATION_DIRECTORY.mkdir(parents=True, exist_ok=True)

## Fixed application file/resource paths
CONFIG_FILE_PATH = APPLICATION_DIRECTORY / "config.json"
LOG_FILE_PATH = APPLICATION_DIRECTORY / "log.txt"
RESOURCES_DIRECTORY = MODULE_DIRECTORY / "resources"
IMAGES_DIRECTORY = RESOURCES_DIRECTORY / "images"
FONT_PATH = RESOURCES_DIRECTORY / "fonts/Rajdhani-Regular.ttf"
SEMIBOLD_FONT_PATH = RESOURCES_DIRECTORY / "fonts/Rajdhani-SemiBold.ttf"

## UI constants
LABEL_HINT_STYLE = "color: #444444;"
BUFFER_ERROR_STYLE = "color: #f04e46;"
SELECTION_OFF_COLOR = "#444444"
MAX_IMAGE_SIZE = 400

## Beep constants
BEEP_START = 800
BEEP_SUCCESS = 1000
BEEP_FAIL = 625
BEEP_DURATION = 100

## Breach protocol data constants
CODE_NAMES = ("1C", "55", "7A", "BD", "E9", "FF")


class Daemon(str, enum.Enum):
    DATAMINE_V1 = "datamine_v1"
    DATAMINE_V2 = "datamine_v2"
    DATAMINE_V3 = "datamine_v3"
    ICEPICK = "icepick"
    MASS_VULNERABILITY = "mass_vulnerability"
    CAMERA_SHUTDOWN = "camera_shutdown"
    FRIENDLY_TURRETS = "friendly_turrets"
    TURRET_SHUTDOWN = "turret_shutdown"
    OPTICS_JAMMER = "optics_jammer"
    WEAPONS_JAMMER = "weapons_jammer"
    DATAMINE_COPY_MALWARE = "datamine_copy_malware"
    NEUTRALIZE_MALWARE = "neutralize_malware"
    GAIN_ACCESS = "gain_access"
    DATAMINE_CRAFTING_SPECS = "datamine_crafting_specs"


DATAMINE_DAEMONS = {
    Daemon.DATAMINE_V1,
    Daemon.DATAMINE_V2,
    Daemon.DATAMINE_V3,
}


COMMON_DAEMONS = DATAMINE_DAEMONS.union(
    {
        Daemon.ICEPICK,
        Daemon.MASS_VULNERABILITY,
        Daemon.CAMERA_SHUTDOWN,
        Daemon.FRIENDLY_TURRETS,
        Daemon.TURRET_SHUTDOWN,
        Daemon.OPTICS_JAMMER,
        Daemon.WEAPONS_JAMMER,
    }
)


class Title(str, enum.Enum):
    BREACH = "breach"
    BUFFER = "buffer"
    SEQUENCES = "sequences"


## Discover available template image languages
TEMPLATE_LANGUAGE_DATA: Dict[str, Dict] = dict()
for _language_directory in (IMAGES_DIRECTORY / "languages").iterdir():
    with (_language_directory / "meta.json").open("rb") as _meta_file:
        _metadata = json.loads(_meta_file.read().decode())
    _metadata["directory"] = _language_directory
    TEMPLATE_LANGUAGE_DATA[_metadata["name"]] = _metadata

## Opencv data parsing constants
ANALYSIS_IMAGE_SIZE = (1920, 1080)
ANALYSIS_IMAGE_RATIO = ANALYSIS_IMAGE_SIZE[0] / ANALYSIS_IMAGE_SIZE[1]
CV_MATRIX_GAP_SIZE = 64.5
CV_BUFFER_BOX_GAP_SIZE = 42.0
CV_SEQUENCES_X_GAP_SIZE = 42.0
CV_SEQUENCES_Y_GAP_SIZE = 70.5
CV_DAEMONS_GAP_SIZE = CV_SEQUENCES_Y_GAP_SIZE
BASE_SEQUENCE_TEMPLATE_WIDTH = 330

## Game window interaction constants
MOUSE_MOVE_DELAY = 0.01
MOUSE_CLICK_DELAY = 0.01

## Matrix constants
VALID_MATRIX_SIZES = tuple(range(4, 9))
MATRIX_IMAGE_FONT_COLOR = (208, 236, 88, 255)
MATRIX_IMAGE_FONT_SIZE = 30
MATRIX_IMAGE_FONT = ImageFont.truetype(str(SEMIBOLD_FONT_PATH), MATRIX_IMAGE_FONT_SIZE)
MATRIX_IMAGE_SPACING = 15
MATRIX_IMAGE_SIZE = 50
MATRIX_TEMPLATE_HALF_SIZE = 15
MATRIX_COMPOSITE_DISTANCE = MATRIX_IMAGE_SIZE + MATRIX_IMAGE_SPACING

## Generated matrix code images
MATRIX_CODE_IMAGES = list()
for _code_name in CODE_NAMES:
    _code_image = Image.new("RGBA", (MATRIX_IMAGE_SIZE,) * 2, color=(0,) * 4)
    _draw = ImageDraw.Draw(_code_image)
    _draw.text(
        (MATRIX_IMAGE_SIZE / 2,) * 2,
        _code_name,
        anchor="mm",
        font=MATRIX_IMAGE_FONT,
        fill=MATRIX_IMAGE_FONT_COLOR,
    )
    MATRIX_CODE_IMAGES.append(_code_image)

## Matrix sequence path overlay constants
SEQUENCE_PATH_IMAGE_COLOR = (95, 247, 255, 255)
SEQUENCE_PATH_IMAGE_INVALID_COLOR = (240, 78, 70, 255)
SEQUENCE_PATH_IMAGE_BOX_SIZE = 40
SEQUENCE_PATH_IMAGE_THICKNESS = 3
SEQUENCE_PATH_IMAGE_ARROW_SIZE = 10
SEQUENCE_PATH_MAX_SIZE = 8

## Buffer box constants
BUFFER_MIN_X_THRESHOLD = 35
BUFFER_COUNT_THRESHOLD = 10
BUFFER_IMAGE_FONT_SIZE = 22
BUFFER_IMAGE_FONT = ImageFont.truetype(str(SEMIBOLD_FONT_PATH), BUFFER_IMAGE_FONT_SIZE)
BUFFER_IMAGE_SPACING = 8
BUFFER_BOX_IMAGE_COLOR = (118, 135, 50)
BUFFER_BOX_IMAGE_SIZE = 36
BUFFER_BOX_IMAGE_THICKNESS = 2
BUFFER_COMPOSITE_DISTANCE = BUFFER_BOX_IMAGE_SIZE + BUFFER_IMAGE_SPACING
MAXIMUM_BUFFER_IMAGE_LENGTH = 344

## Generated buffer boxes and buffer box code images
BUFFER_BOX_IMAGE = Image.new("RGBA", (BUFFER_BOX_IMAGE_SIZE,) * 2, color=(0,) * 4)
_draw = ImageDraw.Draw(BUFFER_BOX_IMAGE)
_draw.rectangle(
    ((0, 0), (BUFFER_BOX_IMAGE_SIZE - 1,) * 2),
    fill=(0,) * 4,
    outline=BUFFER_BOX_IMAGE_COLOR,
    width=BUFFER_BOX_IMAGE_THICKNESS,
)
_quarter_size = int((BUFFER_BOX_IMAGE_SIZE) / 4)
_fifth_size = int((BUFFER_BOX_IMAGE_SIZE) / 5)
_negative_quarter = BUFFER_BOX_IMAGE_SIZE - _quarter_size - 2  ## lol
_lines = (
    ((0, _quarter_size), (BUFFER_BOX_IMAGE_SIZE, _quarter_size)),
    ((0, _negative_quarter), (BUFFER_BOX_IMAGE_SIZE, _negative_quarter)),
)
for _swap in (False, True):
    for _line in _lines:
        _draw.line(
            tuple((_it[::-1] if _swap else _it) for _it in _line),
            fill=(0,) * 4,
            width=_fifth_size,
        )
BUFFER_CODE_IMAGES = list()
for _code_name in CODE_NAMES:
    _code_image = Image.new("RGBA", (BUFFER_BOX_IMAGE_SIZE,) * 2, color=(0,) * 4)
    _draw = ImageDraw.Draw(_code_image)
    _draw.text(
        (BUFFER_BOX_IMAGE_SIZE / 2,) * 2,
        _code_name,
        anchor="mm",
        font=BUFFER_IMAGE_FONT,
        fill=SEQUENCE_PATH_IMAGE_COLOR,
    )
    BUFFER_CODE_IMAGES.append(_code_image)


class Templates:
    def __init__(self):
        self._language: Optional[str] = None

        ## Language agnostic: buffer boxes
        self.buffer_box = _rt(IMAGES_DIRECTORY / "buffer_box.png")

        ## Sometimes language agnostic: codes
        self._codes: Tuple[numpy.ndarray, ...] = tuple()
        self.default_codes: Tuple[numpy.ndarray, ...] = tuple(
            _rt(IMAGES_DIRECTORY / f"code_{it}.png") for it in range(len(CODE_NAMES))
        )
        self._small_codes: Tuple[numpy.ndarray, ...] = tuple()
        self.default_small_codes: Tuple[numpy.ndarray, ...] = tuple(
            _rt(IMAGES_DIRECTORY / f"code_{it}_small.png")
            for it in range(len(CODE_NAMES))
        )
        self._daemons_gap_size: float = CV_DAEMONS_GAP_SIZE

        ## Language dependent: titles and daemons
        self._daemon_names: Dict[Daemon, str] = dict()
        self._daemons: Dict[Daemon, numpy.ndarray] = dict()
        self._titles: Dict[Title, numpy.ndarray] = dict()

    def requires_language(method):
        def _decorated(self, *args, **kwargs):
            if not self._language:
                raise ValueError(
                    f"Templates.{method.__name__} requires a loaded language"
                )
            return method(self, *args, **kwargs)

        return _decorated

    def load_language(self, language: str):
        data = TEMPLATE_LANGUAGE_DATA[language]
        directory = data["directory"]

        if (directory / "code_0.png").exists():
            code_range = range(len(CODE_NAMES))
            self._codes = tuple(_rt(directory / f"code_{it}.png") for it in code_range)
            self._small_codes = tuple(
                _rt(directory / f"code_{it}_small.png") for it in code_range
            )
        else:
            self._codes = self.default_codes
            self._small_codes = self.default_small_codes

        self._daemons_gap_size = data.get("daemons_gap_size", CV_DAEMONS_GAP_SIZE)
        self._daemon_names = {Daemon(k): v for k, v in data["daemons"].items()}
        self._daemons = dict()
        for daemon in Daemon:
            daemon_image_file = directory / f"daemon_{daemon.value}.png"
            if daemon_image_file.exists():
                self._daemons[daemon] = _rt(daemon_image_file)

        self._titles = {it: _rt(directory / f"title_{it.value}.png") for it in Title}
        self._language = language

    @property  # type: ignore
    @requires_language
    def codes(self):
        return self._codes

    @property  # type: ignore
    @requires_language
    def small_codes(self):
        return self._small_codes

    @property  # type: ignore
    @requires_language
    def daemon_names(self):
        return self._daemon_names

    @property  # type: ignore
    @requires_language
    def daemons(self):
        return self._daemons

    @property  # type: ignore
    @requires_language
    def titles(self):
        return self._titles

    @property  # type: ignore
    @requires_language
    def daemons_gap_size(self):
        return self._daemons_gap_size


CV_TEMPLATES = Templates()
