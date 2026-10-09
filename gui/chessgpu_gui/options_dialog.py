"""Engine profiles in the GUI: the list of profiles, one profile's editor,
and the options form built from the engine's own "option" lines (so every
variant the engine offers, e.g. ChessGPU's Search, Model or Eval, can be
chosen without the GUI knowing about it)."""

from __future__ import annotations

import copy
import shlex

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QHBoxLayout,
                               QLabel, QLineEdit, QListWidget, QPushButton, QScrollArea, QSpinBox, QVBoxLayout, QWidget)

from .profiles import EngineProfile
from .uci import UciEngine, UciOption

INT_MAX = 2**31 - 1


class OptionProbe(QObject):
    """Starts an engine just long enough to read its options (until uciok,
    so engines that set up the GPU on isready never get that far)."""

    done = Signal(list, str)  # options, engine name
    failed = Signal(str)

    def __init__(self, profile: EngineProfile, parent: QObject | None = None):
        super().__init__(parent)
        self.profile = profile
        self.engine = UciEngine("option probe", self)
        self.engine.options_received.connect(self._received)
        self.engine.failed.connect(self._failed)

    def start(self) -> None:
        """Call after connecting to done and failed: a missing program fails
        at once, inside this call."""
        self.engine.start(self.profile.path, self.profile.args)

    def _received(self) -> None:
        options, name = list(self.engine.options), self.engine.id_name
        self.engine.quit()
        self.done.emit(options, name)

    def _failed(self, message: str) -> None:
        self.engine.quit()
        self.failed.emit(message)

    def cancel(self) -> None:
        self.engine.quit()


class OptionsForm(QWidget):
    """One input per engine option (buttons excluded), set to the stored
    values or the engine's defaults."""

    def __init__(self, options: list[UciOption], values: dict[str, str], parent: QWidget | None = None):
        super().__init__(parent)
        self.options = [o for o in options if o.type != "button"]
        self.inputs: dict[str, QWidget] = {}
        stored = {k.lower(): v for k, v in values.items()}
        layout = QFormLayout(self)
        for opt in self.options:
            value = stored.get(opt.name.lower(), opt.default)
            widget = self._input(opt, value)
            self.inputs[opt.name] = widget
            layout.addRow(opt.name, widget)
        self.unknown = {k: v for k, v in values.items() if not any(o.name.lower() == k.lower() for o in self.options)}

    @staticmethod
    def _input(opt: UciOption, value: str | None) -> QWidget:
        if opt.type == "check":
            w = QCheckBox()
            w.setChecked((value or "").lower() == "true")
        elif opt.type == "spin" and (opt.min is None or opt.min >= -INT_MAX) and (opt.max is None or opt.max <= INT_MAX):
            w = QSpinBox()
            w.setRange(opt.min if opt.min is not None else -INT_MAX, opt.max if opt.max is not None else INT_MAX)
            try:
                w.setValue(int(value))
            except (TypeError, ValueError):
                pass
        elif opt.type == "combo":
            w = QComboBox()
            w.addItems(opt.vars)
            match = opt.normalize(value) if value is not None else None
            if match is not None:
                w.setCurrentText(match)
        else:  # string, or a spin too wide for a spin box
            w = QLineEdit(value or "")
        return w

    def value(self, name: str) -> str:
        w = self.inputs[name]
        if isinstance(w, QCheckBox):
            return "true" if w.isChecked() else "false"
        if isinstance(w, QSpinBox):
            return str(w.value())
        if isinstance(w, QComboBox):
            return w.currentText()
        return w.text()

    def values(self) -> dict[str, str]:
        """Options whose value differs from the engine's default (plus stored
        values for options the engine no longer offers, kept as they were)."""
        out = dict(self.unknown)
        for opt in self.options:
            v = self.value(opt.name)
            if opt.normalize(v) is None:
                continue
            if opt.default is None or opt.normalize(v) != opt.normalize(opt.default):
                out[opt.name] = opt.normalize(v)
        return out


class ProfileDialog(QDialog):
    """Edits one profile: name, program, arguments and options."""

    def __init__(self, profile: EngineProfile, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle(f"Engine profile: {profile.name}")
        self.profile = copy.deepcopy(profile)
        self.form: OptionsForm | None = None
        self.probe: OptionProbe | None = None
        self.name = QLineEdit(profile.name)
        self.path = QLineEdit(profile.path)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        self.args = QLineEdit(shlex.join(profile.args))
        path_row = QHBoxLayout()
        path_row.addWidget(self.path)
        path_row.addWidget(browse)
        top = QFormLayout()
        top.addRow("Name", self.name)
        top.addRow("Program", path_row)
        top.addRow("Arguments", self.args)
        self.status = QLabel("")
        self.read_button = QPushButton("Read options from the engine")
        self.read_button.clicked.connect(self.read_options)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setMinimumHeight(260)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.read_button)
        layout.addWidget(self.status)
        layout.addWidget(self.scroll)
        layout.addWidget(buttons)
        self._show_stored()
        self.read_options()

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Engine program", self.path.text(),
                                              "Programs (*.exe);;All files (*)")
        if path:
            self.path.setText(path)
            self.read_options()

    def _show_stored(self) -> None:
        text = "\n".join(f"{k} = {v}" for k, v in self.profile.options.items()) or "(engine defaults)"
        label = QLabel(text)
        label.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.scroll.setWidget(label)

    def read_options(self) -> None:
        if self.probe is not None:
            self.probe.cancel()
        self.status.setText("Reading options…")
        self.probe = OptionProbe(EngineProfile("probe", self.path.text().strip(), self._args()), self)
        self.probe.done.connect(self._got_options)
        self.probe.failed.connect(lambda msg: self.status.setText(f"Couldn't read options: {msg}"))
        self.probe.start()

    def _got_options(self, options: list, name: str) -> None:
        values = self.form.values() if self.form is not None else self.profile.options
        self.form = OptionsForm(options, values)
        self.scroll.setWidget(self.form)
        self.status.setText(f"{name or 'engine'}: {len(self.form.options)} options")

    def _args(self) -> list[str]:
        try:
            return shlex.split(self.args.text())
        except ValueError:
            return self.args.text().split()

    def result_profile(self) -> EngineProfile:
        options = self.form.values() if self.form is not None else dict(self.profile.options)
        return EngineProfile(self.name.text().strip() or "engine", self.path.text().strip(), self._args(), options)

    def done(self, code: int) -> None:
        if self.probe is not None:
            self.probe.cancel()
        super().done(code)


class ProfilesDialog(QDialog):
    """The list of engine profiles: add, edit, duplicate, remove."""

    def __init__(self, profiles: list[EngineProfile], parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Engines")
        self.profiles = copy.deepcopy(profiles)
        self.list = QListWidget()
        self.list.itemDoubleClicked.connect(lambda _item: self.edit())
        buttons = QVBoxLayout()
        for text, slot in (("Add…", self.add), ("Edit…", self.edit), ("Duplicate", self.duplicate),
                           ("Remove", self.remove)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            buttons.addWidget(b)
        buttons.addStretch()
        row = QHBoxLayout()
        row.addWidget(self.list)
        row.addLayout(buttons)
        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(row)
        layout.addWidget(box)
        self._refresh()

    def _refresh(self, select: int | None = None) -> None:
        self.list.clear()
        for p in self.profiles:
            extra = ", ".join(f"{k}={v}" for k, v in p.options.items())
            self.list.addItem(p.name + (f"  [{extra}]" if extra else ""))
        if self.profiles:
            self.list.setCurrentRow(min(select if select is not None else 0, len(self.profiles) - 1))

    def _current(self) -> int | None:
        row = self.list.currentRow()
        return row if 0 <= row < len(self.profiles) else None

    def add(self) -> None:
        template = self.profiles[self._current()] if self._current() is not None else EngineProfile("engine", "")
        dialog = ProfileDialog(EngineProfile("new engine", template.path, list(template.args)), self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.profiles.append(dialog.result_profile())
            self._refresh(len(self.profiles) - 1)

    def edit(self) -> None:
        i = self._current()
        if i is None:
            return
        dialog = ProfileDialog(self.profiles[i], self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.profiles[i] = dialog.result_profile()
            self._refresh(i)

    def duplicate(self) -> None:
        i = self._current()
        if i is None:
            return
        twin = copy.deepcopy(self.profiles[i])
        twin.name += " (copy)"
        self.profiles.insert(i + 1, twin)
        self._refresh(i + 1)

    def remove(self) -> None:
        i = self._current()
        if i is not None:
            del self.profiles[i]
            self._refresh(i)
