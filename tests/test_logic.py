import os
import pathlib

from typing import List, Optional, Tuple, Type

import pytest  # type: ignore

import cpah


SCREENSHOTS_DIRECTORY = (
    pathlib.Path(os.path.realpath(__file__)).parent.resolve() / "screenshots"
)


class MatrixTestCase:
    def __init__(
        self,
        test_id: str,
        data: Tuple[Tuple[int, ...], ...],
        buffer_size: int,
        sequences: Tuple[Tuple[int, ...], ...],
        expected_valid: bool,
        expected_solution_length: Optional[int] = None,
        expected_solution_path: Optional[Tuple[Tuple[int, int], ...]] = None,
        expected_number_of_sequence_paths: Optional[int] = None,
        expected_solvable: bool = True,
        expected_selected_indices: Optional[Tuple[int, ...]] = None,
        selected_sequence_indices: Optional[Tuple[int, ...]] = None,
        daemon_priorities: Optional[List[cpah.constants.Daemon]] = None,
        force_autohack: bool = False,
        daemons: Optional[Tuple[cpah.constants.Daemon, ...]] = None,
    ):
        self.test_id = test_id
        self.config = cpah.models.Config(
            daemon_priorities=daemon_priorities or list(),
            force_autohack=force_autohack,
        )
        daemons = daemons or tuple(cpah.constants.Daemon)
        self.breach_protocol_data = cpah.models.BreachProtocolData(
            data=data,
            matrix_size=len(data),
            buffer_size=buffer_size,
            sequences=sequences,
            daemons=daemons[: len(sequences)],
            daemon_names=("",) * len(sequences),
        )
        self.expected_valid = expected_valid
        self.expected_solution_length = expected_solution_length
        self.expected_solution_path = expected_solution_path
        self.expected_number_of_sequence_paths = expected_number_of_sequence_paths
        self.expected_solvable = expected_solvable
        self.expected_selected_indices = expected_selected_indices
        if selected_sequence_indices is None:
            selected_sequence_indices = tuple(range(len(sequences)))
        self.selected_sequence_indices = selected_sequence_indices

    def verify(self):
        if self.config.force_autohack:
            (
                sequence_path_data,
                selected_sequence_indices,
            ) = cpah.logic.force_calculate_sequence_path_data(
                self.config, self.breach_protocol_data
            )
        else:
            sequence_path_data = cpah.logic.calculate_sequence_path_data(
                self.breach_protocol_data, self.selected_sequence_indices
            )
        assert sequence_path_data.solution_valid == self.expected_valid
        if self.expected_solvable:
            assert sequence_path_data.shortest_solution is not None
            if self.expected_number_of_sequence_paths is not None:
                assert (
                    len(sequence_path_data.all_sequence_paths)
                    == self.expected_number_of_sequence_paths
                )
            if self.expected_solution_length is not None:
                assert (
                    len(sequence_path_data.shortest_solution)
                    == self.expected_solution_length
                )
            assert sequence_path_data.shortest_solution_path is not None
            if self.expected_solution_path is not None:
                assert (
                    sequence_path_data.shortest_solution_path
                    == self.expected_solution_path
                )
            if self.expected_selected_indices is not None:
                assert self.expected_selected_indices == selected_sequence_indices
        else:
            assert sequence_path_data.shortest_solution is None
            assert sequence_path_data.shortest_solution_path is None


matrix_test_data = [
    MatrixTestCase(
        test_id="Valid, 6 total nodes",
        data=(
            (0, 1, 0, 0, 0),
            (4, 1, 0, 3, 3),
            (3, 1, 3, 3, 4),
            (4, 1, 1, 1, 1),
            (4, 0, 1, 0, 0),
        ),
        buffer_size=6,
        sequences=((0, 1), (1, 3), (3, 4, 0)),
        expected_valid=True,
        expected_solution_length=6,
    ),
    MatrixTestCase(
        test_id="Invalid, 6 total nodes",
        data=(
            (1, 0, 0, 1, 0),
            (0, 0, 0, 1, 3),
            (1, 3, 4, 1, 0),
            (3, 3, 1, 0, 0),
            (0, 0, 1, 1, 0),
        ),
        buffer_size=5,
        sequences=((0, 3), (1, 0, 0), (0, 0, 1)),
        expected_valid=False,
        expected_solution_length=6,
    ),
    MatrixTestCase(
        test_id="Valid, 1 exploration node",
        data=(
            (0, 1, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 4, 3),
            (0, 2, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0),
        ),
        buffer_size=5,
        sequences=((1, 2), (3, 4)),
        expected_valid=True,
        expected_solution_length=5,
        expected_solution_path=((1, 0), (1, 4), (5, 4), (5, 3), (4, 3)),
    ),
    MatrixTestCase(
        test_id="Valid, 3 exploration nodes",
        data=(
            (0, 1, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0),
            (0, 2, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0),
            (0, 3, 0, 0, 0, 4),
            (0, 0, 0, 0, 0, 0),
        ),
        buffer_size=7,
        sequences=((1, 2), (3, 4)),
        expected_valid=True,
        expected_solution_length=7,
    ),
    MatrixTestCase(
        test_id="Invalid, 3 exploration nodes",
        data=(
            (0, 4, 4, 1, 0, 3),
            (2, 0, 3, 0, 2, 3),
            (2, 0, 2, 0, 3, 2),
            (1, 2, 1, 0, 4, 1),
            (0, 4, 1, 2, 0, 3),
            (1, 1, 0, 0, 0, 1),
        ),
        buffer_size=6,
        sequences=((0, 3, 4), (2, 2, 0), (2, 3, 0)),
        expected_valid=False,
        expected_solution_length=12,
    ),
    MatrixTestCase(
        test_id="Invalid, unsolvable",
        data=(
            (0, 0, 0, 0, 0, 1),
            (0, 0, 0, 0, 3, 2),
            (0, 0, 0, 4, 0, 0),
            (0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0),
        ),
        buffer_size=20,
        sequences=((1, 2), (3, 4)),
        expected_valid=False,
        expected_solvable=False,
    ),
    MatrixTestCase(
        test_id="Valid, sequence combination and splits",
        data=(
            (1, 0, 0, 0, 0, 0),
            (2, 2, 0, 0, 0, 0),
            (0, 0, 2, 0, 0, 0),
            (0, 0, 3, 4, 0, 0),
            (0, 0, 0, 1, 0, 0),
            (0, 0, 0, 0, 0, 0),
        ),
        buffer_size=8,
        sequences=((1, 2, 2), (2, 3), (3, 4, 1)),
        expected_valid=True,
        expected_solution_length=8,
    ),
    MatrixTestCase(
        test_id="Valid, selected indices",
        data=(
            (3, 1, 0, 3, 2, 1),
            (1, 2, 0, 1, 3, 4),
            (4, 2, 1, 3, 0, 2),
            (1, 4, 3, 1, 2, 4),
            (0, 0, 1, 3, 3, 3),
            (0, 0, 0, 1, 4, 2),
        ),
        buffer_size=6,
        sequences=((1, 4, 2), (0, 2, 4), (2, 4, 3, 1)),
        expected_valid=True,
        expected_solution_length=6,
        selected_sequence_indices=(0, 2),
    ),
    MatrixTestCase(
        test_id="Invalid, unsolvable from too many solutions",
        data=(
            (3, 3, 4, 1, 1, 0),
            (0, 4, 1, 0, 4, 0),
            (4, 1, 3, 1, 2, 1),
            (2, 4, 0, 0, 3, 1),
            (3, 2, 2, 2, 1, 3),
            (2, 2, 4, 3, 3, 2),
        ),
        buffer_size=6,
        sequences=((1, 0, 0), (2, 1, 1, 1), (1, 2, 3), (1, 2, 2), (1, 3)),
        expected_valid=False,
        expected_solvable=False,
    ),
    MatrixTestCase(
        test_id="Valid, sequence path permutation consolidation",
        data=(
            (1, 0, 2, 0, 3, 2),
            (2, 0, 1, 0, 0, 1),
            (0, 0, 3, 4, 2, 0),
            (0, 1, 2, 0, 1, 0),
            (0, 0, 0, 4, 2, 2),
            (1, 0, 3, 2, 1, 0),
        ),
        buffer_size=6,
        sequences=((0, 0), (2, 0, 0), (0, 4, 2, 1)),
        expected_valid=True,
        expected_solvable=True,
        expected_number_of_sequence_paths=3,
    ),
    MatrixTestCase(
        test_id="Force autohack, all datamine",
        data=(
            (0, 0, 1, 4, 1),
            (1, 3, 4, 0, 3),
            (0, 3, 1, 4, 4),
            (0, 3, 3, 0, 1),
            (1, 1, 0, 1, 1),
        ),
        buffer_size=4,
        sequences=((1, 3), (3, 1), (1, 0, 1)),
        expected_valid=True,
        expected_solvable=True,
        expected_selected_indices=(0, 2),
        force_autohack=True,
    ),
    MatrixTestCase(
        test_id="Force autohack, custom daemon priorities",
        data=(
            (3, 3, 4, 1, 1, 0),
            (0, 4, 1, 0, 4, 0),
            (4, 1, 3, 1, 2, 1),
            (2, 4, 0, 0, 3, 1),
            (3, 2, 2, 2, 1, 3),
            (2, 2, 4, 3, 3, 2),
        ),
        buffer_size=6,
        sequences=((1, 0, 0), (2, 1, 1, 1), (1, 2, 3), (1, 2, 2), (1, 3)),
        expected_valid=True,
        expected_solvable=True,
        expected_selected_indices=(1, 3),
        force_autohack=True,
        daemon_priorities=[
            cpah.constants.Daemon.FRIENDLY_TURRETS,
            cpah.constants.Daemon.TURRET_SHUTDOWN,
        ],
        daemons=(
            cpah.constants.Daemon.ICEPICK,
            cpah.constants.Daemon.FRIENDLY_TURRETS,
            cpah.constants.Daemon.MASS_VULNERABILITY,
            cpah.constants.Daemon.TURRET_SHUTDOWN,
            cpah.constants.Daemon.CAMERA_SHUTDOWN,
        ),
    ),
]


@pytest.mark.parametrize(
    "matrix_test_case", matrix_test_data, ids=[it.test_id for it in matrix_test_data]
)
def test_sequence_path_calculation(matrix_test_case):
    """
    Runs the sequence path calculation against
    the matrix data and checks the result.
    """
    matrix_test_case.verify()


class ScreenshotTestCase:
    def __init__(
        self,
        test_id: str,
        screenshot_name: str,
        matrix_data: Optional[Tuple[Tuple[int, ...], ...]] = None,
        buffer_size: Optional[int] = None,
        sequences: Optional[Tuple[Tuple[int, ...], ...]] = None,
        daemon_names: Optional[Tuple[str, ...]] = None,
        raises: Optional[Type[Exception]] = None,
        **config_kwargs,
    ):
        self.test_id = test_id
        self.screenshot_name = screenshot_name
        self.matrix_data = matrix_data
        self.buffer_size = buffer_size
        self.sequences = sequences
        self.daemon_names = daemon_names
        self.raises = raises
        self.config = cpah.models.Config(**config_kwargs)

    def _verify(self):
        cpah.constants.CV_TEMPLATES.load_language(self.config.detection_language)
        screenshot_data = cpah.logic.grab_screenshot(
            self.config, SCREENSHOTS_DIRECTORY / self.screenshot_name
        )
        screen_bounds = cpah.logic.parse_screen_bounds(self.config, screenshot_data)

        data = cpah.logic.parse_matrix_data(self.config, screenshot_data, screen_bounds)
        if self.matrix_data is not None:
            assert self.matrix_data == data

        buffer_size = cpah.logic.parse_buffer_size_data(
            self.config, screenshot_data, screen_bounds
        )
        if self.buffer_size is not None:
            assert self.buffer_size == buffer_size

        sequences, daemons, daemon_names = cpah.logic.parse_daemons_data(
            self.config, screenshot_data, screen_bounds
        )
        if self.sequences is not None:
            assert self.sequences == sequences
        if self.daemon_names is not None:
            assert self.daemon_names == daemon_names

    def verify(self):
        """
        Because pytest doesn't seem to support a conditional pytest.raises,
        this wraps the real _verify logic with pytest.raises as necessary.
        """
        if self.raises is None:
            self._verify()
        else:
            with pytest.raises(self.raises):
                self._verify()


screenshot_test_data = [
    ScreenshotTestCase(
        test_id="Standard size, basic, 0.png",
        screenshot_name="0.png",
        matrix_data=(
            (3, 3, 0, 0, 0),
            (4, 1, 0, 1, 1),
            (0, 0, 4, 0, 0),
            (1, 4, 0, 0, 1),
            (1, 1, 0, 0, 1),
        ),
        buffer_size=5,
        sequences=((3, 0), (0, 0), (0, 1)),
        daemon_names=("BASIC DATAMINE", "ADVANCED DATAMINE", "EXPERT DATAMINE"),
    ),
    ScreenshotTestCase(
        test_id="Standard size, prehacked, 1.png",
        screenshot_name="1.png",
        matrix_data=(
            (4, 5, 0, 5, 1, 1, 0),
            (0, 0, 0, 0, 4, 0, 1),
            (0, 0, 0, 4, 3, 3, 1),
            (3, 5, 3, 3, 0, 0, 1),
            (2, 4, 0, 3, 4, 1, 0),
            (3, 5, 0, 2, 0, 1, 2),
            (1, 0, 0, 0, 0, 2, 0),
        ),
        buffer_size=9,
        sequences=((1, 0, 0, 1), (0, 0, 0)),
        daemon_names=("MASS VULNERABILITY", "CAMERA SHUTDOWN"),
    ),
    ScreenshotTestCase(
        test_id="Standard size, invalid cursor position, 2.png",
        screenshot_name="2.png",
        raises=cpah.exceptions.CPAHMatrixParseFailedException,
    ),
    ScreenshotTestCase(
        test_id="Small size, invalid thresholds, 3.png",
        screenshot_name="3.png",
        raises=cpah.exceptions.CPAHBufferParseFailedException,
    ),
    ScreenshotTestCase(
        test_id="Small size, default thresholds, buffer size override, 3.png",
        screenshot_name="3.png",
        matrix_data=(
            (4, 0, 3, 3, 1),
            (3, 3, 0, 1, 1),
            (3, 0, 1, 1, 3),
            (0, 1, 0, 0, 1),
            (1, 3, 4, 3, 0),
        ),
        buffer_size=5,
        sequences=((1, 0), (0, 4), (1, 0)),
        daemon_names=("BASIC DATAMINE", "ADVANCED DATAMINE", "EXPERT DATAMINE"),
        buffer_size_override=5,
    ),
    ScreenshotTestCase(
        test_id="Small size, invalid matrix threshold, 3.png",
        screenshot_name="3.png",
        matrix_code_detection_threshold=0.1,
        raises=cpah.exceptions.CPAHMatrixParseFailedException,
    ),
    ScreenshotTestCase(
        test_id="Small size, invalid sequence threshold, buffer size override, 3.png",
        screenshot_name="3.png",
        buffer_size_override=6,
        sequence_code_detection_threshold=0.1,
        raises=cpah.exceptions.CPAHSequenceParseFailedException,
    ),
    ScreenshotTestCase(
        test_id="Medium size, invalid buffer threshold, 4.png",
        screenshot_name="4.png",
        raises=cpah.exceptions.CPAHBufferParseFailedException,
    ),
    ScreenshotTestCase(
        test_id="Medium size, valid buffer threshold, 4.png",
        screenshot_name="4.png",
        buffer_box_detection_threshold=0.5,
        matrix_data=(
            (1, 0, 3, 0, 3),
            (1, 0, 3, 1, 1),
            (3, 4, 0, 3, 0),
            (0, 1, 0, 0, 3),
            (3, 4, 4, 1, 4),
        ),
        buffer_size=5,
        sequences=((4, 1), (3, 0), (4, 4)),
        daemon_names=("BASIC DATAMINE", "ADVANCED DATAMINE", "EXPERT DATAMINE"),
    ),
    ScreenshotTestCase(
        test_id="Standard size, several daemons, 5.png",
        screenshot_name="5.png",
        matrix_data=(
            (4, 0, 4, 3, 0, 4),
            (0, 2, 0, 1, 3, 1),
            (2, 0, 0, 1, 1, 0),
            (0, 3, 0, 1, 1, 0),
            (0, 1, 2, 1, 0, 0),
            (3, 3, 0, 0, 1, 2),
        ),
        buffer_size=6,
        sequences=((0, 4, 2), (3, 0, 1), (1, 0, 0), (0, 0, 3, 0), (2, 1, 3)),
        daemon_names=(
            "ICEPICK",
            "MASS VULNERABILITY",
            "TURRET SHUTDOWN",
            "FRIENDLY TURRETS",
            "CAMERA SHUTDOWN",
        ),
    ),
    ScreenshotTestCase(
        test_id="Invalid screenshot, 6.png",
        screenshot_name="6.png",
        raises=cpah.exceptions.CPAHScreenshotParseFailedException,
    ),
    ScreenshotTestCase(
        test_id="Ultrawide size, 7.png",
        screenshot_name="7.png",
        matrix_data=(
            (0, 3, 4, 1, 5, 4, 4),
            (3, 1, 3, 3, 1, 3, 2),
            (1, 1, 3, 3, 5, 0, 3),
            (0, 5, 0, 1, 0, 1, 0),
            (4, 2, 0, 4, 5, 0, 0),
            (2, 1, 1, 3, 5, 2, 0),
            (2, 3, 0, 2, 3, 5, 5),
        ),
        buffer_size=8,
        sequences=((1, 0, 2), (3, 3, 1), (1, 3, 0, 0)),
        daemon_names=("BASIC DATAMINE", "ADVANCED DATAMINE", "EXPERT DATAMINE"),
    ),
    ScreenshotTestCase(
        test_id="Standard size, Simplified Chinese, 8.png",
        screenshot_name="8.png",
        matrix_data=(
            (4, 1, 0, 1, 0),
            (1, 0, 4, 1, 3),
            (0, 4, 3, 0, 0),
            (3, 3, 0, 3, 0),
            (0, 0, 1, 0, 4),
        ),
        buffer_size=5,
        sequences=(
            (1, 0),
            (3, 1),
            (4, 3),
        ),
        daemon_names=(
            "数据挖掘_V1",
            "数据挖掘_V2",
            "数据挖掘_V3",
        ),
        detection_language="\u7b80\u4f53\u4e2d\u6587",
    ),
    ScreenshotTestCase(
        test_id="Standard size, 9 buffer size, prehacked, 9.png",
        screenshot_name="9.png",
        matrix_data=(
            (1, 2, 0, 3, 0, 0, 0),
            (4, 4, 1, 3, 1, 1, 5),
            (0, 1, 1, 0, 1, 0, 2),
            (1, 1, 1, 0, 2, 0, 0),
            (5, 1, 0, 5, 1, 0, 5),
            (2, 2, 2, 1, 1, 1, 1),
            (4, 1, 1, 2, 2, 5, 1),
        ),
        buffer_size=9,
        sequences=((1, 5, 2, 2),),
        daemon_names=("MASS VULNERABILITY",),
    ),
    ScreenshotTestCase(
        test_id="Small size, stretched aspect ratio, 10.png",
        screenshot_name="10.png",
        buffer_box_detection_threshold=0.5,
        matrix_data=(
            (1, 1, 3, 0, 3, 1),
            (3, 2, 2, 4, 0, 2),
            (1, 3, 0, 2, 3, 3),
            (3, 0, 1, 4, 2, 4),
            (0, 2, 3, 2, 0, 2),
            (3, 3, 2, 0, 1, 0),
        ),
        buffer_size=6,
        sequences=((2, 3, 2, 1), (3, 0, 3), (0, 2, 4), (0, 1, 2, 0), (4, 2)),
        daemon_names=(
            "ICEPICK",
            "MASS VULNERABILITY",
            "TURRET SHUTDOWN",
            "FRIENDLY TURRETS",
            "CAMERA SHUTDOWN",
        ),
    ),
]


@pytest.mark.parametrize(
    "screenshot_test_case",
    screenshot_test_data,
    ids=[it.test_id for it in screenshot_test_data],
)
def test_screenshot_parsing(screenshot_test_case):
    """Runs the screenshot parsing logic and checks the result."""
    screenshot_test_case.verify()
