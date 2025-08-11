import dataclasses
import functools
import os
import pathlib
import sys
import time
import types
import webbrowser
import winsound

from typing import Callable, Dict, Iterable, List, Optional, Tuple, Type, Union

import pyautogui  # type: ignore
pyautogui.FAILSAFE = False
import system_hotkey  # type: ignore

from PIL import Image  # type: ignore
from PySide2.QtCore import QFile, QObject, QRect, Qt, QThread, Signal, QPoint
from PySide2.QtGui import (
    QCloseEvent,
    QFontDatabase,
    QHideEvent,
    QIcon,
    QPixmap,
    QShowEvent,
)
from PySide2.QtUiTools import QUiLoader
from PySide2.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QDesktopWidget,
)

from . import constants
from . import exceptions
from . import logic
from . import models

from .logger import LOG


## Workaround for certain unhandled exceptions using PyQt5 instead of PySide2
## And also ImageQt for using PyQt5
## SO: 27340162
import PySide2

sys.modules["pyQt5"] = PySide2
from PIL import ImageQt


class Worker(QObject):
    started = Signal()
    finished = Signal()
    error = Signal(models.Error)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._args = tuple()
        self._kwargs = dict()

    def run(self, *args, **kwargs):
        raise NotImplementedError

    def wrapped_run(self):
        """Long-running task."""
        LOG.debug(f"Worker thread started: {type(self).__name__}")
        self.started.emit()
        try:
            self.run(*self._args, **self._kwargs)
        except Exception as exception:
            LOG.exception("Worker thread encountered an exception:")
            self.error.emit(models.Error(exception))
        LOG.debug(f"Worker thread finished: {type(self).__name__}")
        self.finished.emit()


@dataclasses.dataclass
class ThreadContainer:
    thread: QThread
    worker: Worker


def _beep_sequence(config: models.Config, frequencies: Iterable[int]):
    """Helper function for generating and playing notification beeps."""
    if config.enable_beeps:
        for frequency in frequencies:
            winsound.Beep(frequency, constants.BEEP_DURATION)


class SHATimeoutWorker(Worker):
    """
    A worker that simply waits the given amount of time.
    Used for sequential hotkey action timeout checking.
    """

    def run(self, duration: float):  # type: ignore
        time.sleep(duration)


class SolverWorker(Worker):
    matrix_overlay_image_signal = Signal(Image.Image)
    buffer_overlay_image_signal = Signal(Image.Image)
    solution_signal = Signal(models.SequencePathData)
    sequence_selection_signal = Signal(tuple)
    sequence_path_data_available_signal = Signal(models.SequencePathData)
    solver_finished_signal = Signal(bool)

    def run(  # type: ignore
        self,
        config: models.Config,
        breach_protocol_data: models.BreachProtocolData,
        selected_sequence_indices: Tuple[int],
        force_solve: bool = False,
        hotkey: bool = False,
    ):
        self._solve_and_signal(
            config,
            breach_protocol_data,
            selected_sequence_indices=selected_sequence_indices,
            force_solve=force_solve,
        )
        self.solver_finished_signal.emit(hotkey)  # type: ignore

    def _solve_and_signal(
        self,
        config: models.Config,
        breach_protocol_data: models.BreachProtocolData,
        selected_sequence_indices: Optional[Tuple[int, ...]] = None,
        force_solve: bool = False,
        beep: bool = False,
    ) -> models.SequencePathData:
        """
        Convenience function for solving the matrix,
        generating the images, and emitting the proper signals on the worker.
        """
        if force_solve:
            (
                sequence_path_data,
                selected_sequence_indices,
            ) = logic.force_calculate_sequence_path_data(config, breach_protocol_data)
            beep = config.enable_beeps
        else:
            if selected_sequence_indices is None:
                selected_sequence_indices = tuple(
                    range(len(breach_protocol_data.sequences))
                )
            sequence_path_data = logic.calculate_sequence_path_data(
                breach_protocol_data,
                selected_sequence_indices=selected_sequence_indices,
            )
        self.sequence_selection_signal.emit(selected_sequence_indices)  # type: ignore
        matrix_overlay_image = None
        if sequence_path_data.shortest_solution_path:
            image_size = (
                constants.MATRIX_IMAGE_SIZE * breach_protocol_data.matrix_size
                + constants.MATRIX_IMAGE_SPACING
                * (breach_protocol_data.matrix_size - 1)
            )
            matrix_overlay_image = logic.generate_sequence_path_image(
                image_size,
                sequence_path_data.shortest_solution_path,
                sequence_path_data.solution_valid,
            )
            self.matrix_overlay_image_signal.emit(matrix_overlay_image)  # type: ignore
        if sequence_path_data.solution_valid:
            buffer_sequence_image = logic.generate_buffer_sequence_image(
                breach_protocol_data.buffer_size,
                sequence_path_data.shortest_solution,  # type: ignore
            )
            self.buffer_overlay_image_signal.emit(buffer_sequence_image)  # type: ignore
            self.sequence_path_data_available_signal.emit(sequence_path_data)  # type: ignore
        self.solution_signal.emit(sequence_path_data)  # type: ignore
        if beep:
            beep_tones = [constants.BEEP_START]
            if force_solve:
                base_beeps = [constants.BEEP_FAIL] * len(breach_protocol_data.daemons)
                for selected_index in selected_sequence_indices:
                    base_beeps[selected_index] = constants.BEEP_SUCCESS
                beep_tones.extend(base_beeps)
            else:
                beep_tones.append(
                    constants.BEEP_SUCCESS
                    if sequence_path_data.solution_valid
                    else constants.BEEP_FAIL
                )
            _beep_sequence(config, beep_tones)
        return sequence_path_data


class AnalysisWorker(SolverWorker):
    status = Signal(str)
    matrix_image_signal = Signal(Image.Image)
    buffer_image_signal = Signal(Image.Image)
    sequences_signal = Signal(tuple)
    analysis_data_available_signal = Signal(models.BreachProtocolData)
    run_autohack_signal = Signal()
    analysis_finished_signal = Signal(bool, bool)

    def run(  # type: ignore
        self,
        config: models.Config,
        from_file: Optional[pathlib.Path] = None,
        hotkey: bool = False,
    ):
        self.status.emit("SCREEN")  # type: ignore
        screenshot_data = logic.grab_screenshot(config, from_file=from_file)

        self.status.emit("DATA POSITIONS")  # type: ignore
        screen_bounds = logic.parse_screen_bounds(config, screenshot_data)

        self.status.emit("CODE MATRIX")  # type: ignore
        data = logic.parse_matrix_data(config, screenshot_data, screen_bounds)
        image_size = constants.MATRIX_IMAGE_SIZE * len(
            data
        ) + constants.MATRIX_IMAGE_SPACING * (len(data) - 1)
        matrix_image = logic.generate_matrix_image(image_size, data)
        self.matrix_image_signal.emit(matrix_image)  # type: ignore

        self.status.emit("BUFFER SIZE")  # type: ignore
        buffer_size = logic.parse_buffer_size_data(
            config, screenshot_data, screen_bounds
        )
        buffer_boxes_image = logic.generate_buffer_boxes_image(buffer_size)
        self.buffer_image_signal.emit(buffer_boxes_image)  # type: ignore

        self.status.emit("DAEMONS")  # type: ignore
        sequences, daemons, daemon_names = logic.parse_daemons_data(
            config, screenshot_data, screen_bounds
        )
        sequence_selection_data = [
            models.SequenceSelectionData(
                daemon=daemon,
                daemon_name=daemon_name,
                sequence=sequence,
                selected=False,
            )
            for daemon, daemon_name, sequence in zip(daemons, daemon_names, sequences)
        ]
        self.sequences_signal.emit(sequence_selection_data)  # type: ignore
        breach_protocol_data = models.BreachProtocolData(
            data=data,
            matrix_size=len(data),
            buffer_size=buffer_size,
            sequences=sequences,
            daemons=daemons,
            daemon_names=daemon_names,
        )
        analysis_data = models.AnalysisData(
            breach_protocol_data=breach_protocol_data,
            screenshot_data=screenshot_data,
            screen_bounds=screen_bounds,
        )
        self.analysis_data_available_signal.emit(analysis_data)  # type: ignore

        self.status.emit("SOLUTIONS")  # type: ignore
        sequence_path_data = self._solve_and_signal(
            config,
            breach_protocol_data,
            force_solve=config.force_autohack,
            beep=config.enable_beeps,
        )

        run_autohack = (
            sequence_path_data.solution_valid and config.auto_autohack and not from_file
        )
        if run_autohack:
            self.run_autohack_signal.emit()  # type: ignore
        self.analysis_finished_signal.emit(run_autohack, hotkey)  # type: ignore


class AutohackWorker(Worker):
    def run(  # type: ignore
        self,
        config: models.Config,
        breach_protocol_data: models.BreachProtocolData,
        screenshot_data: models.ScreenshotData,
        sequence_path_data: models.SequencePathData,
    ):
        _beep_sequence(config, [constants.BEEP_START])
        logic.autohack(
            config, breach_protocol_data, screenshot_data, sequence_path_data
        )
        _beep_sequence(config, [constants.BEEP_SUCCESS])


class ErrorHandlerMixin:
    """Mixin to add exception handling."""

    _error_handler_signal = Signal(models.Error, QWidget)

    def __init__(self, *args, **kwargs):
        super(ErrorHandlerMixin, self).__init__(*args, **kwargs)
        self.error_box = QMessageBox()
        self._error_handler_signal.connect(self.show_error_message)

    def _handle_exceptions(method):
        """Adds exception handling to the method."""
        self_focus = getattr(method, "_EHM_self_focus", True)

        @functools.wraps(method)
        def _decorated(self, *args, **kwargs):
            try:
                return method(self, *args, **kwargs)
            except Exception as exception:
                LOG.exception("Exception handler caught exception:")
                self._error_handler_signal.emit(
                    models.Error(exception), self if self_focus else None
                )

        _decorated._EHM_nonce = True
        return _decorated

    @staticmethod
    def class_decorator(cls):
        """Decorates all functions of the widget to have error handling"""
        for attr in cls.__dict__:
            method = getattr(cls, attr)
            if (
                isinstance(method, types.FunctionType)
                and not getattr(method, "_EHM_nonce", False)
                and not getattr(method, "_EHM_ignore", False)
            ):
                setattr(cls, attr, ErrorHandlerMixin._handle_exceptions(method))
        return cls

    @staticmethod
    def ignore(method):
        """Decorator modifier to ignore error handling by the mixin."""
        method._EHM_ignore = True
        return method

    @staticmethod
    def no_self_focus(method):
        """Decorator modifier to disable self-focus on error."""
        method._EHM_self_focus = False
        return method

    def show_error_message(self, error: models.Error, focus: Optional[QWidget] = None):
        LOG.error(f"Displaying error message: {error}")
        if error.critical:
            button, dialog = QMessageBox.Close, self.error_box.critical
        else:
            button, dialog = QMessageBox.Ok, self.error_box.warning
        dialog(focus, error.title, error.message, button)
        if error.critical:
            LOG.info("Closing application due to critical error")
            ## Attempting to exit with QApplication.exit causes a stack overflow. Cool.
            sys.exit(2)


@ErrorHandlerMixin.class_decorator
class CPAH(ErrorHandlerMixin, QWidget):
    analysis_hotkey_signal = Signal()
    autohack_hotkey_signal = Signal()
    daemon_toggle_hotkey_signal = Signal(int)

    def __init__(self):
        super().__init__()

        ## Internal
        self.center()
        self.oldPos = self.pos()
        self.autohack_warning_box = QMessageBox()
        self.screenshot_picker_dialog = QFileDialog()
        self._recycled_threads: Dict[str, ThreadContainer] = dict()
        self._temp_threads: List[ThreadContainer] = list()
        self.hotkey = system_hotkey.SystemHotkey()

        ## This nested decorator and monkey patching abomination is to
        ## wrap errors that are raised from the hotkey registering thread
        ## while also retaining information about what hotkey was attempted
        ## to be registered.
        def _error_handle_wrap_hotkey(method):
            _original_grab = self.hotkey._the_grab

            def _decorated(*args, **kwargs):
                def _sh_wrap(inner_method):
                    def _inner_decorated(*inner_args, **inner_kwargs):
                        try:
                            return inner_method(*inner_args, **inner_kwargs)
                        except system_hotkey.SystemRegisterError as exception:
                            self._error_handler_signal.emit(
                                models.Error(
                                    exceptions.CPAHHotkeyRegistrationException(
                                        exception,
                                        full_bind=args[0],
                                    )
                                ),
                                self,
                            )

                    return _inner_decorated

                self.hotkey._the_grab = _sh_wrap(_original_grab)
                return method(*args, **kwargs)

            return _decorated

        ## Patch system_hotkey so that errors are properly caught when using Qt
        self.hotkey.register = _error_handle_wrap_hotkey(self.hotkey.register)

        ## State
        self.analyzing = False
        self.autohacking = False
        self.configuration_screen_open = False
        self.hotkey_action_timeout = 0.0

        ## Cache
        self.analysis_data: models.AnalysisData = None
        self.sequence_path_data: models.SequencePathData = None
        self.buffer_image_cache: Image.Image = None

        self.config = logic.load_config()
        self.load_ui()
        self.bootstrap()

    @ErrorHandlerMixin.ignore
    def _set_up_worker(
        self, name: str, worker: Type[Worker], recycle: bool = True
    ) -> Tuple[Callable, ThreadContainer]:
        """Convenience function for setting up a worker thread."""
        thread_container = self._recycled_threads.get(name)
        if thread_container:
            try:
                if thread_container.thread.isRunning():
                    raise exceptions.CPAHThreadRunningException
            except RuntimeError:  ## Thread already deleted
                pass

        thread_container = ThreadContainer(QThread(), worker())
        if recycle:
            self._recycled_threads[name] = thread_container
        else:
            to_remove: List[int] = list()
            for index, temp_container in enumerate(self._temp_threads):
                try:
                    if not temp_container.thread.isRunning():
                        to_remove.append(index)
                except RuntimeError:
                    to_remove.append(index)
            for it in to_remove[::-1]:
                del self._temp_threads[it]
            self._temp_threads.append(thread_container)
        thread_container.worker.moveToThread(thread_container.thread)
        thread_container.thread.started.connect(thread_container.worker.wrapped_run)
        thread_container.thread.finished.connect(thread_container.thread.deleteLater)
        thread_container.worker.finished.connect(thread_container.thread.quit)  # type: ignore
        thread_container.worker.finished.connect(thread_container.worker.deleteLater)  # type: ignore
        thread_container.worker.error.connect(self.show_error_message)  # type: ignore

        def _start_worker(*args, **kwargs):
            thread_container.worker._args = args
            thread_container.worker._kwargs = kwargs
            thread_container.thread.start()

        return _start_worker, thread_container

    @ErrorHandlerMixin.ignore
    def show_error_message(self, error: models.Error, focus: Optional[QWidget] = None):
        super().show_error_message(error, focus=focus or self)
        LOG.info("Resetting displays due to error")
        self.reset_displays()
        self.analyzing = False
        self.autohacking = False

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def configuration_screen_open_changed(self, state: bool):
        LOG.debug(f"Configuration screen open: {state}")
        self.configuration_screen_open = state

    def cache_analysis_data(self, data: models.AnalysisData):
        LOG.debug(f"Caching analysis data: {data}")
        self.analysis_data = data

    def cache_sequence_path_data(self, data: models.SequencePathData):
        LOG.debug(f"Caching sequence path data: {data}")
        self.sequence_path_data = data

    def start_screenshot_analysis(self):
        """
        Like start_analysis, but uses an existing
        screenshot instead of taking a new one.
        """
        LOG.debug("start_screenshot_analysis")
        self.screenshot_picker_dialog.setFileMode(QFileDialog.ExistingFile)
        self.screenshot_picker_dialog.setNameFilter("Images (*.png *.jpg)")
        if self.screenshot_picker_dialog.exec_():
            selected_file = self.screenshot_picker_dialog.selectedFiles()[0]
            LOG.debug(f"User selected screenshot: {selected_file}")
            self.start_analysis(from_file=pathlib.Path(selected_file))

    def _in_hotkey_sequence(self) -> bool:
        """Helper function for determining if the hotkey sequence is active."""
        return bool(
            self.config.sequential_hotkey_actions
            and not self.config.auto_autohack
            and self.hotkey_action_timeout
            and time.time() < self.hotkey_action_timeout
            and self.sequence_path_data
        )

    def _refresh_hotkey_sequence(self):
        """Helper function for setting/refreshing the hotkey sequence state."""
        self.hotkey_action_timeout = (
            time.time() + self.config.sequential_hotkey_actions_timeout / 1000
        )
        self.sequential_hotkey_action_timeout_checker()

    def analysis_hotkey_pressed(self):
        LOG.debug("analysis_hotkey_pressed")
        if self._in_hotkey_sequence():
            if self.sequence_path_data.solution_valid:
                LOG.debug("Sequential hotkey action: autohack")
                self.start_autohack()
            else:
                LOG.debug("Sequential hotkey action: deselect daemons")
                self.recalculate_solution(force_solve=True, hotkey=True)
        else:
            self.start_analysis(hotkey=True)

    def autohack_hotkey_pressed(self):
        LOG.debug("autohack_hotkey_pressed")
        self.start_autohack()

    def start_analysis(
        self, from_file: Optional[pathlib.Path] = None, hotkey: bool = False
    ):
        LOG.debug("start_analysis")
        if self.configuration_screen_open:
            LOG.debug("Ignoring analysis start because the config screen is open.")
            return
        elif self.autohacking:
            LOG.debug("Ignoring analysis start because autohacking is in progress.")
            return
        elif self.analyzing:
            LOG.debug("Ignoring analysis start because analysis is still running.")
            return
        try:
            start_worker, thread_container = self._set_up_worker(
                "analysis", AnalysisWorker
            )
        except exceptions.CPAHThreadRunningException:
            LOG.warning("Analysis worker already running! Ignoring start...")
            return

        self.analyzing = True

        ## Clear all images and cache
        self.reset_displays()
        self.analysis_data = None  # type: ignore
        self.sequence_path_data = None  # type: ignore

        ## Disable the analyze button until it's done
        self.analyze_button.setEnabled(False)

        _beep_sequence(self.config, [constants.BEEP_START])

        thread_container.worker.matrix_image_signal.connect(self.set_matrix_image)
        thread_container.worker.matrix_overlay_image_signal.connect(
            self.set_matrix_overlay_image
        )
        thread_container.worker.buffer_image_signal.connect(self.set_buffer_image)
        thread_container.worker.buffer_overlay_image_signal.connect(
            self.set_buffer_overlay_image
        )
        thread_container.worker.status.connect(self.show_analyzing_status)
        thread_container.worker.analysis_finished_signal.connect(self.analysis_finished)
        thread_container.worker.sequences_signal.connect(
            self.sequence_container.set_sequences
        )
        thread_container.worker.solution_signal.connect(self.show_solution)
        thread_container.worker.analysis_data_available_signal.connect(
            self.cache_analysis_data
        )
        thread_container.worker.sequence_selection_signal.connect(
            self.sequence_container.select_sequences
        )
        thread_container.worker.sequence_path_data_available_signal.connect(
            self.cache_sequence_path_data
        )
        thread_container.worker.run_autohack_signal.connect(self.start_autohack)
        start_worker(self.config, from_file=from_file, hotkey=hotkey)

    def analysis_finished(self, autohacking: bool = False, hotkey_action: bool = False):
        LOG.debug("analysis_finished")
        if hotkey_action:
            self._refresh_hotkey_sequence()
        self.analyzing = False
        self.analyze_button.setEnabled(not autohacking)
        self._reset_analyze_button_text()

    def sequential_hotkey_action_timeout_checker(self):
        """
        An uncreatively named helper function for checking if the
        sequential hotkey action has timed out.
        """
        LOG.debug("sequential_hotkey_action_timeout_checker")
        if (
            not self.config.auto_autohack
            and self.config.sequential_hotkey_actions
            and self.hotkey_action_timeout
        ):
            remaining_time = self.hotkey_action_timeout - time.time()
            if remaining_time <= 0:
                self.hotkey_action_timeout = 0.0
                self._reset_analyze_button_text()
                _beep_sequence(self.config, [constants.BEEP_FAIL] * 2)
            else:
                start_worker, thread_container = self._set_up_worker(
                    "sha_timeout", SHATimeoutWorker, recycle=False
                )
                thread_container.worker.finished.connect(
                    self.sequential_hotkey_action_timeout_checker
                )
                start_worker(remaining_time)
                LOG.debug(f"Starting SHATimeoutWorker with duration: {remaining_time}")
        else:
            self.hotkey_action_timeout = 0.0
        self._reset_analyze_button_text()

    def show_analyzing_status(self, status: str):
        LOG.debug(f"Analyzing status set to: {status}")
        self.analyze_button.setText(f"ANALYZING... [{status}]")

    def start_autohack(self) -> None:
        LOG.debug("start_autohack")

        if not (self.sequence_path_data and self.sequence_path_data.solution_valid):
            LOG.debug(
                "Ignoring autohack start because the "
                "sequence path data does not have a solution"
            )
            return
        elif not self._get_enabled_selections():
            LOG.debug("Ignoring autohack start because there are no selected daemons")
            return
        elif self.autohacking:
            LOG.debug("Ignoring autohack start because an autohack is already running")
            return

        if self._in_hotkey_sequence():
            self.hotkey_action_timeout = 0.0
            self.sequential_hotkey_action_timeout_checker()

        if self.config.show_autohack_warning_message:
            LOG.debug("Showing first time autohack warning")
            self.autohack_warning_box.warning(
                self,
                "Autohack warning",
                (
                    "When running the autohack, it is important that you do not move "
                    "the mouse until the autohack is finished, because that will "
                    "reset the code matrix cursor position and will cause the "
                    "autohack to fail!\n\n"
                    "Also note that if Cyberpunk is running in admin mode, CPAH will "
                    "also need to be running in admin mode in order for autohacking "
                    "to work. If autohacking doesn't do anything, try launching CPAH "
                    "in admin mode.\n\n"
                    "This message will only be displayed once."
                ),
                QMessageBox.Ok,
            )
            self.config.show_autohack_warning_message = False
            logic.save_config(self.config)

        try:
            start_worker, thread_container = self._set_up_worker(
                "autohack", AutohackWorker
            )
        except exceptions.CPAHThreadRunningException:
            LOG.warning("Autohacker worker already running! Ignoring autohack...")
            return

        self.autohacking = True
        self.analyze_button.setEnabled(False)
        self.autohack_button.setEnabled(False)
        self.autohack_button.setText(f"正在自动破解...")
        thread_container.worker.finished.connect(self.autohack_finished)
        start_worker(
            self.config,
            self.analysis_data.breach_protocol_data,
            self.analysis_data.screenshot_data,
            self.sequence_path_data,
        )

    def autohack_finished(self):
        LOG.debug("autohack_finished")
        self.autohacking = False
        self.analyze_button.setEnabled(True)
        self.autohack_button.setEnabled(True)
        self.autohack_button.setText(f"自动破解")

    def daemon_toggle_hotkey_pressed(self, daemon_index):
        LOG.debug(f"daemon_toggle_hotkey_pressed (daemon_index={daemon_index})")
        if self._in_hotkey_sequence():
            self._refresh_hotkey_sequence()
        if daemon_index < len(self.sequence_container.sequences):
            self.sequence_container.sequences[daemon_index].toggle()

    def open_configuration(self):
        LOG.debug("open_configuration")
        self.configuration_screen.show()

    def configuration_changed(self, reset_displays: bool = True):
        LOG.debug("Resetting displays due to configuration")

        pyautogui.PAUSE = self.config.autohack_keypress_delay / 1000
        LOG.debug(f"Set pyautogui.PAUSE value to {pyautogui.PAUSE}")

        LOG.debug(f"Setting detection language to {self.config.detection_language}")
        constants.CV_TEMPLATES.load_language(self.config.detection_language)

        existing_binds = tuple(self.hotkey.keybinds)
        for existing_bind in existing_binds:
            LOG.debug(f"Unregistering bind {existing_bind}")
            self.hotkey.unregister(existing_bind)

        try:
            if self.config.analysis_hotkey:
                LOG.debug(f"Registering bind {self.config.analysis_hotkey}")
                self.hotkey.register(
                    self.config.analysis_hotkey,
                    callback=lambda _: self.analysis_hotkey_signal.emit(),  # type: ignore
                )
            if self.config.autohack_hotkey:
                LOG.debug(f"Registering bind {self.config.autohack_hotkey}")
                self.hotkey.register(
                    self.config.autohack_hotkey,
                    callback=lambda _: self.autohack_hotkey_signal.emit(),  # type: ignore
                )
            if self.config.daemon_toggle_hotkey:
                LOG.debug(
                    "Registering daemon toggling binds for "
                    f"{self.config.daemon_toggle_hotkey} 1-8"
                )

                def _emit_wrapper(*args, index):
                    self.daemon_toggle_hotkey_signal.emit(index)

                for daemon_index in range(8):
                    bind = self.config.daemon_toggle_hotkey + [f"{daemon_index + 1}"]
                    LOG.debug(f"Registering daemon toggle bind: {bind}")
                    _callback = functools.partial(_emit_wrapper, index=daemon_index)
                    self.hotkey.register(bind, callback=_callback)
        except system_hotkey.SystemRegisterError as register_error:
            LOG.exception("Hotkey could not be registered")
            self._error_handler_signal.emit(  # type: ignore
                models.Error(
                    exceptions.CPAHHotkeyRegistrationException(register_error)
                ),
                self,
            )

        if reset_displays:
            self.reset_displays()

    def _get_enabled_selections(self) -> List[int]:
        """Helper for getting enabled daemon selections."""
        enabled_selections = list()
        for index, sequence_selection in enumerate(self.sequence_container.sequences):
            if sequence_selection.data.selected:
                enabled_selections.append(index)
        return enabled_selections

    def show_solution(self, sequence_path_data: models.SequencePathData):
        LOG.debug("show_solution")
        self.cache_sequence_path_data(sequence_path_data)
        if sequence_path_data.solution_valid:
            self.autohack_button.setEnabled(True)
        else:
            self.autohack_button.setEnabled(False)
            self.show_buffer_error_message(sequence_path_data)

    def recalculate_solution(self, force_solve: bool = False, hotkey: bool = False):
        LOG.debug(f"recalculate_solution (force_solve={force_solve})")
        try:
            start_worker, thread_container = self._set_up_worker("solver", SolverWorker)
        except exceptions.CPAHThreadRunningException:
            LOG.warning("Solver worker already running! Ignoring recalculation...")
            return

        thread_container.worker.solution_signal.connect(self.show_solution)
        thread_container.worker.matrix_overlay_image_signal.connect(
            self.set_matrix_overlay_image
        )
        thread_container.worker.buffer_overlay_image_signal.connect(
            self.set_buffer_overlay_image
        )
        thread_container.worker.sequence_selection_signal.connect(
            self.sequence_container.select_sequences
        )
        thread_container.worker.sequence_path_data_available_signal.connect(
            self.cache_sequence_path_data
        )
        thread_container.worker.solver_finished_signal.connect(
            self.recalculate_solution_finished
        )

        ## All entries deselected
        enabled_selections = self._get_enabled_selections()
        if not enabled_selections:
            self.set_matrix_overlay_image()
            self.set_buffer_overlay_image()
            self.sequence_container.select_sequences(tuple())
            self.autohack_button.setEnabled(False)
        else:
            start_worker(
                self.config,
                self.analysis_data.breach_protocol_data,
                tuple(enabled_selections),
                force_solve=force_solve,
                hotkey=hotkey,
            )

    def recalculate_solution_finished(self, hotkey_action: bool):
        LOG.debug("recalculate_solution_finished")
        if hotkey_action:
            self._refresh_hotkey_sequence()

    def _convert_and_set_image(self, image: Image.Image, label: QLabel):
        """Helper function to convert an image and set it on the label."""
        qt_image = ImageQt.ImageQt(image)
        pixmap = QPixmap.fromImage(qt_image)
        label.setPixmap(pixmap)

    def set_matrix_image(self, image: Optional[Image.Image] = None):
        LOG.debug("Setting matrix image")
        if image is None:
            self.matrix_label.setPixmap(None)
            self.set_matrix_overlay_image()
            self.matrix_overlay_label.setStyleSheet(constants.LABEL_HINT_STYLE)
            self.matrix_overlay_label.setText("代码矩阵")
        else:
            self.matrix_overlay_label.setText("")
            self._convert_and_set_image(image, self.matrix_label)

    def set_matrix_overlay_image(self, image: Optional[Image.Image] = None):
        LOG.debug("Setting matrix overlay image")
        if image is None:
            self.matrix_overlay_label.setPixmap(None)
        else:
            self._convert_and_set_image(image, self.matrix_overlay_label)

    def set_buffer_image(self, image: Optional[Image.Image] = None):
        LOG.debug("Setting buffer image")
        if image is None:
            self.buffer_label.setPixmap(None)
            self.set_buffer_overlay_image()
            self.buffer_overlay_label.setStyleSheet(constants.LABEL_HINT_STYLE)
            self.buffer_overlay_label.setText("缓冲区")
        else:
            self.buffer_image_cache = image
            self.buffer_overlay_label.setText("")
            self._convert_and_set_image(image, self.buffer_label)

    def set_buffer_overlay_image(self, image: Optional[Image.Image] = None):
        LOG.debug("Setting buffer overlay image")
        if image is None:
            self.buffer_overlay_label.setPixmap(None)
        else:
            self.set_buffer_image(self.buffer_image_cache)
            self._convert_and_set_image(image, self.buffer_overlay_label)

    def show_buffer_error_message(self, sequence_path_data: models.SequencePathData):
        LOG.debug("Setting buffer error message")
        tooltip: Optional[str] = None
        if sequence_path_data.shortest_solution:
            keep_path_overlay = True
            tooltip = "  ".join(
                logic.convert_code(sequence_path_data.shortest_solution)
            )
            length = len(sequence_path_data.shortest_solution)
            message = f"SEQUENCE TOO LONG ({length})"
        else:
            keep_path_overlay = False
            if sequence_path_data.computationally_complex:
                message = "SEQUENCE TOO LONG"
            else:
                message = "NO VALID PATH FOUND"

        if not keep_path_overlay:
            self.set_matrix_overlay_image()
        self.set_buffer_image()
        self.buffer_overlay_label.setStyleSheet(constants.BUFFER_ERROR_STYLE)
        self.buffer_overlay_label.setText(message)
        self.buffer_overlay_label.setToolTip(tooltip)

    def _reset_analyze_button_text(self):
        """Helper function to set the analyze button text."""
        text = "分析"
        if self.config.auto_autohack:
            text += " + 自动破解"
        elif (
            self.config.sequential_hotkey_actions
            and self.hotkey_action_timeout
            and time.time() < self.hotkey_action_timeout
        ):
            text += "  (HOTKEY -> AUTOHACK)"

        self.analyze_button.setText(text)

    def reset_displays(self):
        """Resets the images/messages."""
        LOG.debug("Resetting displays")
        self._reset_analyze_button_text()
        self.analyze_button.setEnabled(True)
        self.autohack_button.setEnabled(False)
        self.set_matrix_image()
        self.set_buffer_image()
        self.sequence_container.set_sequences(None)
        self.analysis_data = None
        self.sequence_path_data = None
        self.buffer_image_cache = None

    def bootstrap(self):
        """Loads the config file and sets the system-wide autohack hotkey."""
        LOG.debug("Bootstrapping application")
        self.analysis_hotkey_signal.connect(self.analysis_hotkey_pressed)
        self.autohack_hotkey_signal.connect(self.autohack_hotkey_pressed)
        self.daemon_toggle_hotkey_signal.connect(self.daemon_toggle_hotkey_pressed)
        self.configuration_changed(reset_displays=False)

    def mousePressEvent(self, event):
        self.oldPos = event.globalPos()

    def mouseMoveEvent(self, event):
        delta = QPoint(event.globalPos() - self.oldPos)
        self.move(self.x() + delta.x(), self.y() + delta.y())
        self.oldPos = event.globalPos()

    def load_ui(self):
        loader = QUiLoader()
        ui_file = QFile(str(constants.RESOURCES_DIRECTORY / "uis/main.ui"))
        ui_file.open(QFile.ReadOnly)
        loader.load(ui_file, self)
        ui_file.close()

        ## Window properties
        self.setWindowIcon(QIcon(str(constants.IMAGES_DIRECTORY / "icon.ico")))
        self.setWindowTitle(f"{constants.APPLICATION_NAME} - {constants.VERSION}")
        self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)

        # self.setWindowFlags(self.windowFlags() | Qt.MSWindowsFixedSizeDialogHint)

        ## Load Title and Close Pixmap
        self.titleLabel = self.findChild(QLabel, "titleLabel")
        self.titleLabel.setPixmap(
            QPixmap(str(constants.IMAGES_DIRECTORY / "title.png"))
        )
        self.titleLabel.setAttribute(Qt.WA_TranslucentBackground, True)

        self.closeButton = self.findChild(QPushButton, "closeButton")
        self.closeButton.setIcon(QIcon(str(constants.IMAGES_DIRECTORY / "close.png")))
        self.closeButton.setAttribute(Qt.WA_TranslucentBackground, True)

        ## Set Version Label
        self.versionLabel = self.findChild(QLabel, "versionLabel")
        self.versionLabel.setText(f"v{constants.VERSION}")

        ## Font loading
        QFontDatabase.addApplicationFont(str(constants.FONT_PATH))

        ## Connect buttons
        self.closeButton.clicked.connect(self.close)
        self.analyze_button = self.findChild(QPushButton, "analyzeButton")
        self.analyze_button.clicked.connect(self.start_analysis)
        self.analyze_button.setContextMenuPolicy(Qt.CustomContextMenu)
        self.analyze_button.customContextMenuRequested.connect(
            lambda _: self.start_screenshot_analysis()
        )
        self.autohack_button = self.findChild(QPushButton, "autohackButton")
        self.autohack_button.clicked.connect(self.start_autohack)
        self.configure_button = self.findChild(QPushButton, "configureButton")
        self.configure_button.clicked.connect(self.open_configuration)

        ## Labels to be referenced later for displaying images
        self.matrix_label = self.findChild(QLabel, "matrixLabel")
        self.matrix_overlay_label = self.findChild(QLabel, "matrixOverlayLabel")
        self.buffer_label = self.findChild(QLabel, "bufferLabel")
        self.buffer_overlay_label = self.findChild(QLabel, "bufferOverlayLabel")
        self.sequences_label = self.findChild(QLabel, "sequencesLabel")

        self.matrix_overlay_label.setAttribute(Qt.WA_TranslucentBackground, True)

        ## Load SequenceContainer into the scroll area
        self.sequence_container = SequenceContainer(self)
        self.scroll_area_widget = self.findChild(QScrollArea, "sequenceScrollArea")
        self.scroll_area_widget.setWidget(self.sequence_container)

        ## Load configuration screen
        self.configuration_screen = ConfigurationScreen(self)
        self.configuration_screen.configuration_changed_signal.connect(
            self.configuration_changed
        )
        self.configuration_screen.configuration_screen_open_signal.connect(
            self.configuration_screen_open_changed
        )

        self.reset_displays()


@ErrorHandlerMixin.class_decorator
class SequenceContainer(ErrorHandlerMixin, QWidget):
    """Container that holds all available SequenceSelections."""

    @ErrorHandlerMixin.no_self_focus
    def __init__(self, parent_widget: CPAH):
        super().__init__()
        self.parent_widget = parent_widget
        self.sequences: List[SequenceSelection] = list()
        layout = QVBoxLayout()
        self.setLayout(layout)

    @ErrorHandlerMixin.no_self_focus
    def set_sequences(self, sequences: Optional[List[models.SequenceSelectionData]]):
        layout = self.layout()
        for old_sequence in self.sequences:
            layout.removeWidget(old_sequence)
            old_sequence.deleteLater()

        self.sequences = list()
        for sequence_index, sequence_selection_data in enumerate(sequences or list()):
            selection = SequenceSelection(
                self.parent_widget.config, sequence_index, sequence_selection_data
            )
            selection.selection_updated.connect(self.parent_widget.recalculate_solution)
            scroll_callback = functools.partial(self.scroll_sequence_to_view, selection)
            selection.selection_updated.connect(scroll_callback)
            layout.addWidget(selection)
            self.sequences.append(selection)
        calculated_height = 76 * len(self.sequences) - (
            12 * max(len(self.sequences) - 1, 0)
        )
        LOG.debug(f"Calculated sequence display height: {calculated_height}")
        self.setGeometry(QRect(0, 0, 367, calculated_height))
        self.parent_widget.sequences_label.setText(
            "" if self.sequences else "序列"
        )

    @ErrorHandlerMixin.no_self_focus
    def scroll_sequence_to_view(self, sequence_selection: "SequenceSelection"):
        self.parent_widget.scroll_area_widget.ensureWidgetVisible(sequence_selection)

    @ErrorHandlerMixin.no_self_focus
    def select_sequences(self, selections: Tuple[int, ...]):
        LOG.debug(f"Selecting sequences: {selections}")
        for index, sequence_selection in enumerate(self.sequences):
            sequence_selection.set_selected(index in selections)


@ErrorHandlerMixin.class_decorator
class SequenceSelection(ErrorHandlerMixin, QFrame):
    """A sequence option that shows the sequence and reward type."""

    selection_updated = Signal()

    @ErrorHandlerMixin.no_self_focus
    def __init__(
        self,
        config: models.Config,
        sequence_index: int,
        sequence_selection_data: models.SequenceSelectionData,
    ):
        super().__init__()
        self.data = sequence_selection_data
        self.setCursor(Qt.PointingHandCursor)
        sequence_text = "  ".join(logic.convert_code(self.data.sequence))
        self.sequence_label = QLabel(self)
        self.sequence_label.setText(sequence_text)
        self.sequence_label.setGeometry(QRect(10, 30, 350, 20))
        self.daemon_label = QLabel(self)
        label_name = self.data.daemon_name
        if config.daemon_toggle_hotkey:
            label_name = f"{sequence_index + 1}: {label_name}"
        self.daemon_label.setText(label_name)
        self.daemon_label.setGeometry(QRect(10, 10, 350, 20))
        self.update_appearance()

    @ErrorHandlerMixin.no_self_focus
    def set_selected(self, selected: bool):
        self.data.selected = selected
        self.update_appearance()

    @ErrorHandlerMixin.no_self_focus
    def update_appearance(self):
        color = "white" if self.data.selected else constants.SELECTION_OFF_COLOR
        self.setStyleSheet(f"color: {color};\nborder-color: {color};")

    @ErrorHandlerMixin.no_self_focus
    def toggle(self):
        self.data.selected = not self.data.selected
        self.selection_updated.emit()

    @ErrorHandlerMixin.no_self_focus
    def mouseReleaseEvent(self, event):
        self.toggle()
        ## Appearance is later updated via the container calling set_selected


@ErrorHandlerMixin.class_decorator
class ConfigurationScreen(ErrorHandlerMixin, QWidget):
    configuration_screen_open_signal = Signal(bool)
    configuration_changed_signal = Signal()

    _keep_mapping = {
        constants.Daemon.DATAMINE_V1: "keepDatamine1CheckBox",
        constants.Daemon.DATAMINE_V2: "keepDatamine2CheckBox",
        constants.Daemon.DATAMINE_V3: "keepDatamine3CheckBox",
        constants.Daemon.ICEPICK: "keepIcepickCheckBox",
        constants.Daemon.MASS_VULNERABILITY: "keepMassVulnerabilityCheckBox",
        constants.Daemon.CAMERA_SHUTDOWN: "keepCameraShutdownCheckBox",
        constants.Daemon.TURRET_SHUTDOWN: "keepTurretShutdownCheckBox",
        constants.Daemon.FRIENDLY_TURRETS: "keepFriendlyTurretsCheckBox",
        constants.Daemon.OPTICS_JAMMER: "keepOpticsJammerCheckBox",
        constants.Daemon.WEAPONS_JAMMER: "keepWeaponsJammerCheckBox",
    }

    def __init__(self, parent_widget: CPAH):
        super().__init__()
        self.parent_widget = parent_widget
        self.load_ui()

    def load_ui(self):
        loader = QUiLoader()
        ui_file = QFile(str(constants.RESOURCES_DIRECTORY / "uis/settings.ui"))
        ui_file.open(QFile.ReadOnly)
        loader.load(ui_file, self)
        ui_file.close()

        ## Window properties
        self.setWindowTitle("Configuration")
        self.setWindowFlags(self.windowFlags() | Qt.MSWindowsFixedSizeDialogHint)
        self.setWindowModality(Qt.ApplicationModal)

        ## Interface settings
        self.automatic_autohacking_check_box = self.findChild(
            QCheckBox, "automaticAutohackingCheckBox"
        )
        self.beep_notifications_check_box = self.findChild(
            QCheckBox, "beepNotificationsCheckBox"
        )
        self.analysis_hotkey_line_edit = self.findChild(
            QLineEdit, "analysisHotkeyLineEdit"
        )
        self.autohack_hotkey_line_edit = self.findChild(
            QLineEdit, "autohackHotkeyLineEdit"
        )
        self.focus_delay_spin_box = self.findChild(QSpinBox, "focusDelaySpinBox")

        ## Detection settings
        self.buffer_size_override_spin_box = self.findChild(
            QSpinBox, "bufferSizeOverrideSpinBox"
        )
        self.core_text_spin_box = self.findChild(QDoubleSpinBox, "coreTextSpinBox")
        self.matrix_codes_spin_box = self.findChild(
            QDoubleSpinBox, "matrixCodesSpinBox"
        )
        self.daemon_sequence_spin_box = self.findChild(
            QDoubleSpinBox, "daemonSequenceSpinBox"
        )
        self.buffer_boxes_spin_box = self.findChild(
            QDoubleSpinBox, "bufferBoxesSpinBox"
        )
        self.daemon_names_spin_box = self.findChild(
            QDoubleSpinBox, "daemonNamesSpinBox"
        )
        self.detection_language_combo_box = self.findChild(
            QComboBox, "detectionLanguageComboBox"
        )

        ## Autohack settings
        self.force_autohack_check_box = self.findChild(
            QCheckBox, "forceAutohackCheckBox"
        )
        self.keep_daemons: Dict[constants.Daemon, QCheckBox] = dict()
        for daemon_enum, widget_id in self._keep_mapping.items():
            self.keep_daemons[daemon_enum] = self.findChild(QCheckBox, widget_id)
        self.activation_key_line_edit = self.findChild(
            QLineEdit, "activationKeyLineEdit"
        )
        self.up_key_line_edit = self.findChild(QLineEdit, "upKeyLineEdit")
        self.down_key_line_edit = self.findChild(QLineEdit, "downKeyLineEdit")
        self.left_key_line_edit = self.findChild(QLineEdit, "leftKeyLineEdit")
        self.right_key_line_edit = self.findChild(QLineEdit, "rightKeyLineEdit")
        self.autohack_keypress_delay_spin_box = self.findChild(
            QSpinBox, "autohackKeypressDelaySpinBox"
        )

        ## Advanced settings
        self.window_title_line_edit = self.findChild(QLineEdit, "windowTitleLineEdit")
        self.ahk_check_box = self.findChild(QCheckBox, "ahkCheckBox")
        self.ahk_executable_line_edit = self.findChild(
            QLineEdit, "ahkExecutableLineEdit"
        )
        self.ahk_executable_browse_button = self.findChild(
            QPushButton, "ahkExecutableBrowseButton"
        )
        self.ahk_activation_key_line_edit = self.findChild(
            QLineEdit, "ahkActivationKeyLineEdit"
        )
        self.ahk_up_key_line_edit = self.findChild(QLineEdit, "ahkUpKeyLineEdit")
        self.ahk_down_key_line_edit = self.findChild(QLineEdit, "ahkDownKeyLineEdit")
        self.ahk_left_key_line_edit = self.findChild(QLineEdit, "ahkLeftKeyLineEdit")
        self.ahk_right_key_line_edit = self.findChild(QLineEdit, "ahkRightKeyLineEdit")
        self.sha_check_box = self.findChild(QCheckBox, "shaCheckBox")
        self.sha_timeout_spin_box = self.findChild(QSpinBox, "shaTimeoutSpinBox")
        self.daemon_toggle_hotkey_line_edit = self.findChild(
            QLineEdit, "daemonToggleHotkeyLineEdit"
        )

        ## Connect buttons
        self.ahk_executable_browse_button.clicked.connect(
            self.open_ahk_executable_file_picker
        )
        self.documentation_button = self.findChild(QPushButton, "documentationButton")
        self.documentation_button.clicked.connect(
            lambda: webbrowser.open(constants.DOCUMENTATION_LINK)
        )
        self.open_logs_button = self.findChild(QPushButton, "openLogsButton")
        self.open_logs_button.clicked.connect(
            lambda: os.startfile(constants.APPLICATION_DIRECTORY)
        )
        self.ok_button = self.findChild(QPushButton, "okButton")
        self.ok_button.clicked.connect(lambda: self.close(save=True))
        self.cancel_button = self.findChild(QPushButton, "cancelButton")
        self.cancel_button.clicked.connect(self.close)

    def open_ahk_executable_file_picker(self):
        result = QFileDialog.getOpenFileName(
            self, caption="Select AutoHotkey.exe", filter="Executable Files (*.exe)"
        )
        LOG.debug(f"File picker selected ahk executable: {result}")
        if result and result[0]:
            self.ahk_executable_line_edit.setText(result[0])

    def show(self):
        self.fill_values()
        super().show()

    def fill_values(self):
        """Fills in the configuration fields from loaded config values."""
        config = self.parent_widget.config

        ## Interface settings
        self.automatic_autohacking_check_box.setChecked(config.auto_autohack)
        self.beep_notifications_check_box.setChecked(config.enable_beeps)
        self.analysis_hotkey_line_edit.setText(" + ".join(config.analysis_hotkey))
        self.autohack_hotkey_line_edit.setText(" + ".join(config.autohack_hotkey))
        self.focus_delay_spin_box.setValue(config.game_focus_delay)

        ## Detection settings
        self.buffer_size_override_spin_box.setValue(config.buffer_size_override)
        self.core_text_spin_box.setValue(config.core_detection_threshold)
        self.matrix_codes_spin_box.setValue(config.matrix_code_detection_threshold)
        self.daemon_sequence_spin_box.setValue(config.sequence_code_detection_threshold)
        self.buffer_boxes_spin_box.setValue(config.buffer_box_detection_threshold)
        self.daemon_names_spin_box.setValue(config.daemon_detection_threshold)
        self.detection_language_combo_box.clear()
        self.detection_language_combo_box.addItems(
            constants.TEMPLATE_LANGUAGE_DATA.keys()
        )
        self.detection_language_combo_box.setCurrentIndex(
            list(constants.TEMPLATE_LANGUAGE_DATA).index(config.detection_language)
        )

        ## Autohack settings
        self.force_autohack_check_box.setChecked(config.force_autohack)
        for daemon_enum, keep_daemon in self.keep_daemons.items():
            keep_daemon.setChecked(daemon_enum in config.daemon_priorities)
        key_names = ("activation", "up", "down", "left", "right")
        for key_name in key_names:
            getattr(self, f"{key_name}_key_line_edit").setText(
                getattr(config.autohack_key_bindings, key_name)
            )
        self.autohack_keypress_delay_spin_box.setValue(config.autohack_keypress_delay)

        ## Advanced settings
        self.window_title_line_edit.setText(config.window_title)
        self.ahk_check_box.setChecked(config.ahk_enabled)
        self.ahk_executable_line_edit.setText(config.ahk_executable)
        for key_name in key_names:
            getattr(self, f"ahk_{key_name}_key_line_edit").setText(
                getattr(config.ahk_autohack_key_bindings, key_name)
            )
        self.sha_check_box.setChecked(config.sequential_hotkey_actions)
        self.sha_timeout_spin_box.setValue(config.sequential_hotkey_actions_timeout)
        self.daemon_toggle_hotkey_line_edit.setText(
            " + ".join(config.daemon_toggle_hotkey)
        )

    def showEvent(self, event: QShowEvent):
        self.configuration_screen_open_signal.emit(True)  # type: ignore

    def hideEvent(self, event: QHideEvent):
        self.configuration_screen_open_signal.emit(False)  # type: ignore

    def closeEvent(self, event: QCloseEvent):
        self.close()

    def close(self, save: bool = False):
        LOG.debug(f"Closing configuration window (save={save})")

        if save:
            test = self.parent_widget.config.copy(deep=True)

            ## Interface settings
            test.auto_autohack = self.automatic_autohacking_check_box.isChecked()
            test.enable_beeps = self.beep_notifications_check_box.isChecked()
            hotkey_string = self.analysis_hotkey_line_edit.text().lower().strip()
            if hotkey_string:
                hotkeys = list(it.strip() for it in hotkey_string.split("+"))
            else:
                hotkeys = list()
            test.analysis_hotkey = hotkeys
            hotkey_string = self.autohack_hotkey_line_edit.text().lower().strip()
            if hotkey_string:
                hotkeys = list(it.strip() for it in hotkey_string.split("+"))
            else:
                hotkeys = list()
            test.autohack_hotkey = hotkeys
            test.game_focus_delay = self.focus_delay_spin_box.value()

            ## Detection settings
            test.buffer_size_override = self.buffer_size_override_spin_box.value()
            test.core_detection_threshold = self.core_text_spin_box.value()
            test.matrix_code_detection_threshold = self.matrix_codes_spin_box.value()
            test.sequence_code_detection_threshold = (
                self.daemon_sequence_spin_box.value()
            )
            test.buffer_box_detection_threshold = self.buffer_boxes_spin_box.value()
            test.daemon_detection_threshold = self.daemon_names_spin_box.value()
            selected_language = self.detection_language_combo_box.currentText()
            test.detection_language = selected_language

            ## Autohack settings
            test.force_autohack = self.force_autohack_check_box.isChecked()
            test.daemon_priorities = [
                k for k, v in self.keep_daemons.items() if v.isChecked()
            ]
            key_names = ("activation", "up", "down", "left", "right")
            for key_name in key_names:
                key_value = getattr(self, f"{key_name}_key_line_edit")
                key_value = key_value.text().lower().strip()
                setattr(test.autohack_key_bindings, key_name, key_value)
            test.autohack_keypress_delay = self.autohack_keypress_delay_spin_box.value()

            ## Advanced settings
            test.ahk_enabled = self.ahk_check_box.isChecked()
            test.ahk_executable = self.ahk_executable_line_edit.text().strip()
            test.window_title = self.window_title_line_edit.text().strip()
            for key_name in key_names:
                key_value = getattr(self, f"ahk_{key_name}_key_line_edit")
                key_value = key_value.text().strip()
                setattr(test.ahk_autohack_key_bindings, key_name, key_value)
            test.sequential_hotkey_actions = self.sha_check_box.isChecked()
            test.sequential_hotkey_actions_timeout = self.sha_timeout_spin_box.value()
            hotkey_string = self.daemon_toggle_hotkey_line_edit.text().lower().strip()
            if hotkey_string:
                hotkeys = list(it.strip() for it in hotkey_string.split("+"))
            else:
                hotkeys = list()
            test.daemon_toggle_hotkey = hotkeys

            test_dict = test.dict()
            LOG.debug(f"Testing validity of config: {test_dict}")
            try:
                test.validate(test_dict)
                if test.ahk_enabled:
                    logic._get_ahk_client(test)
            except (ValueError, exceptions.CPAHAHKException) as exception:
                raise exceptions.CPAHInvalidNewConfigException(exception)

            self.parent_widget.config = test
            logic.save_config(self.parent_widget.config)
            self.configuration_changed_signal.emit()  # type: ignore

        self.hide()


def start():
    LOG.info(f"Starting {constants.APPLICATION_NAME} {constants.VERSION}")
    LOG.debug(f"Module directory: {constants.MODULE_DIRECTORY}")
    constants.QAPPLICATION_INSTANCE.setAttribute(Qt.AA_EnableHighDpiScaling)
    try:
        widget = CPAH()
    except Exception as exception:
        LOG.exception("Application startup exception:")
        error = models.Error(exception)
        QMessageBox().critical(None, error.title, error.message, QMessageBox.Close)
        sys.exit(1)
    widget.show()
    sys.exit(constants.QAPPLICATION_INSTANCE.exec_())
