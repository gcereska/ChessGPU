"""UCI: parsing what engines print, and UciEngine, a QProcess-based client.

The client never confuses searches. Every go() gets an id; a new search
starts only after the previous one's bestmove has arrived (the client stops
the old search first), and the info and bestmove signals carry the id of the
search they belong to, so callers simply ignore ids they no longer want.
Commands that UCI forbids during a search (setoption, ucinewgame) wait in the
same queue until the engine is idle.
"""

from __future__ import annotations

import enum
import os
import re
from dataclasses import dataclass, field

from PySide6.QtCore import QObject, QProcess, QTimer, Signal

from .score import Score

UCIOK_TIMEOUT_MS = 10_000
READYOK_TIMEOUT_MS = 60_000  # isready may set up the GPU search
QUIT_TIMEOUT_MS = 3_000

_OPTION_KEYWORDS = ("name", "type", "default", "min", "max", "var")
_INFO_INTS = {"depth": "depth", "seldepth": "seldepth", "multipv": "multipv", "nodes": "nodes", "nps": "nps",
              "time": "time_ms", "hashfull": "hashfull", "tbhits": "tbhits", "currmovenumber": "currmovenumber"}
_MOVE_RE = re.compile(r"^[a-h][1-8][a-h][1-8][qrbn]?$|^0000$")


@dataclass
class UciOption:
    """One "option" line: name, type (check, spin, combo, button, string)."""

    name: str
    type: str
    default: str | None = None
    min: int | None = None
    max: int | None = None
    vars: list[str] = field(default_factory=list)

    def normalize(self, value) -> str | None:
        """The value as the engine expects it, or None if it isn't valid."""
        text = str(value).strip()
        if self.type == "check":
            low = text.lower()
            return low if low in ("true", "false") else None
        if self.type == "spin":
            try:
                n = int(text)
            except ValueError:
                return None
            if (self.min is not None and n < self.min) or (self.max is not None and n > self.max):
                return None
            return str(n)
        if self.type == "combo":
            for v in self.vars:
                if v.lower() == text.lower():
                    return v
            return None
        if self.type == "button":
            return ""
        return text  # string


def parse_option(line: str) -> UciOption | None:
    """Reads "option name <name> type <type> [default ...] [min ...] [max ...]
    [var ...]..."; names and values may contain spaces. None if malformed."""
    words = line.split()
    if len(words) < 2 or words[0] != "option":
        return None
    fields: dict[str, list[str]] = {}
    vars_: list[list[str]] = []
    key = None
    for w in words[1:]:
        # A name runs until "type"; other values until the next keyword.
        if (key == "name" and w == "type") or (key != "name" and w in _OPTION_KEYWORDS):
            key = w
            if w == "var":
                vars_.append([])
            else:
                fields[w] = []
            continue
        if key is None:
            return None
        (vars_[-1] if key == "var" else fields[key]).append(w)
    if not fields.get("name") or not fields.get("type"):
        return None

    def number(k):
        try:
            return int(fields[k][0]) if fields.get(k) else None
        except ValueError:
            return None

    default = " ".join(fields["default"]) if "default" in fields else None
    if default == "<empty>":
        default = ""
    return UciOption(" ".join(fields["name"]), fields["type"][0].lower(), default, number("min"), number("max"),
                     [" ".join(v) for v in vars_])


@dataclass
class Info:
    """One "info" line. Fields the engine didn't send stay None (pv: empty)."""

    depth: int | None = None
    seldepth: int | None = None
    multipv: int | None = None
    score: Score | None = None
    bound: str | None = None  # "lower" or "upper" when the score is only a bound
    nodes: int | None = None
    nps: int | None = None
    time_ms: int | None = None
    hashfull: int | None = None
    tbhits: int | None = None
    currmove: str | None = None
    currmovenumber: int | None = None
    pv: list[str] = field(default_factory=list)
    string: str | None = None


def parse_info(line: str) -> Info:
    words = line.split()
    info = Info()
    i = 1 if words and words[0] == "info" else 0
    while i < len(words):
        w = words[i]
        if w in _INFO_INTS and i + 1 < len(words):
            try:
                setattr(info, _INFO_INTS[w], int(words[i + 1]))
            except ValueError:
                pass
            i += 2
        elif w == "score" and i + 2 < len(words) and words[i + 1] in ("cp", "mate"):
            try:
                value = int(words[i + 2])
                info.score = Score(cp=value) if words[i + 1] == "cp" else Score(mate=value)
            except ValueError:
                pass
            i += 3
            while i < len(words) and words[i] in ("lowerbound", "upperbound"):
                info.bound = words[i][:5]
                i += 1
        elif w == "wdl":
            i += 4
        elif w == "currmove" and i + 1 < len(words):
            info.currmove = words[i + 1]
            i += 2
        elif w in ("pv", "refutation", "currline"):
            i += 1
            moves = []
            while i < len(words) and (_MOVE_RE.match(words[i]) or (w == "currline" and words[i].isdigit())):
                moves.append(words[i])
                i += 1
            if w == "pv":
                info.pv = moves
        elif w == "string":
            info.string = " ".join(words[i + 1:])
            break
        else:
            i += 1
    return info


def parse_bestmove(line: str) -> tuple[str | None, str | None]:
    """(best, ponder) from "bestmove <move> [ponder <move>]"; best is None for
    "(none)" or "0000" (no legal move)."""
    words = line.split()
    best = words[1] if len(words) > 1 else None
    if best in ("(none)", "0000", "none", "null"):
        best = None
    ponder = words[3] if len(words) > 3 and words[2] == "ponder" else None
    return best, ponder


def go_command(limits: dict) -> str:
    """"go ..." from limits such as wtime=1000, movetime=500, depth=8, infinite=True."""
    parts = ["go"]
    for key in ("wtime", "btime", "winc", "binc", "movestogo", "depth", "nodes", "movetime"):
        value = limits.get(key)
        if value is not None:
            parts += [key, str(int(value))]
    if limits.get("infinite"):
        parts.append("infinite")
    return " ".join(parts)


class EngineState(enum.Enum):
    STOPPED = "stopped"  # no process
    STARTING = "starting"  # waiting for uciok, then readyok
    IDLE = "idle"
    SEARCHING = "searching"
    QUITTING = "quitting"
    FAILED = "failed"


class UciEngine(QObject):
    options_received = Signal()  # "uciok": id and options are known (before setoption and isready)
    ready = Signal()  # the handshake finished (first readyok)
    info = Signal(int, object)  # search id, Info
    bestmove = Signal(int, object, object)  # search id, best move (str or None), ponder (str or None)
    log = Signal(str, str)  # ">" sent, "<" received, "!" event; the line
    failed = Signal(str)  # the engine died, couldn't start or broke the protocol
    state_changed = Signal(object)  # EngineState

    def __init__(self, name: str = "engine", parent: QObject | None = None,
                 uciok_timeout_ms: int = UCIOK_TIMEOUT_MS, readyok_timeout_ms: int = READYOK_TIMEOUT_MS):
        super().__init__(parent)
        self.name = name
        self.id_name = ""
        self.id_author = ""
        self.options: list[UciOption] = []
        self.state = EngineState.STOPPED
        self._uciok_timeout_ms = uciok_timeout_ms
        self._readyok_timeout_ms = readyok_timeout_ms
        self._proc: QProcess | None = None
        self._buffer = b""
        self._next_id = 0
        self._active: int | None = None  # the search the engine is running (go sent, no bestmove yet)
        self._stopping = False  # "stop" was sent for the active search
        self._queue: list[tuple] = []  # ("cmd", text) or ("search", id, position, go): waiting for idle
        self._startup_options: dict[str, str] = {}
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._on_timeout)
        self._waiting_for = ""  # "uciok" or "readyok" while the timer runs

    # --- Lifecycle -------------------------------------------------------------

    def start(self, program: str, args: list[str] | None = None, options: dict | None = None) -> None:
        """Launches the engine; `ready` follows once it answered uci and isready."""
        if self._proc is not None:
            raise RuntimeError(f"{self.name} already started")
        self._startup_options = dict(options or {})
        self._proc = QProcess(self)
        self._proc.setProgram(program)
        self._proc.setArguments(list(args or []))
        folder = os.path.dirname(os.path.abspath(program))
        if os.path.isdir(folder):
            self._proc.setWorkingDirectory(folder)
        self._proc.readyReadStandardOutput.connect(self._read_stdout)
        self._proc.readyReadStandardError.connect(self._read_stderr)
        self._proc.errorOccurred.connect(self._on_process_error)
        self._proc.finished.connect(self._on_finished)
        self._proc.started.connect(self._on_started)
        self._set_state(EngineState.STARTING)
        self._event(f"starting {program} {' '.join(args or [])}".rstrip())
        self._proc.start()

    def quit(self) -> None:
        """Asks the engine to quit and waits a little; kills it if it doesn't."""
        if self._proc is None:
            return
        proc = self._proc
        self._queue.clear()
        self._timer.stop()
        if self.state not in (EngineState.FAILED, EngineState.STOPPED):
            self._set_state(EngineState.QUITTING)
            if proc.state() == QProcess.ProcessState.Running:
                if self._active is not None:
                    self._send("stop")
                self._send("quit")
                if not proc.waitForFinished(QUIT_TIMEOUT_MS):
                    self._event("no exit after quit: killing")
                    proc.kill()
                    proc.waitForFinished(QUIT_TIMEOUT_MS)
        elif proc.state() != QProcess.ProcessState.NotRunning:
            proc.kill()
            proc.waitForFinished(QUIT_TIMEOUT_MS)
        self._proc = None
        self._active = None
        self._set_state(EngineState.STOPPED)

    @property
    def running(self) -> bool:
        return self.state in (EngineState.IDLE, EngineState.SEARCHING)

    @property
    def searching(self) -> bool:
        return self._active is not None

    # --- Commands --------------------------------------------------------------

    def option(self, name: str) -> UciOption | None:
        for o in self.options:
            if o.name.lower() == name.lower():
                return o
        return None

    def set_option(self, name: str, value) -> bool:
        """Queues "setoption" (sent once the engine is idle). False if the
        engine has no such option or the value isn't valid for it."""
        opt = self.option(name)
        if opt is None:
            return False
        text = opt.normalize(value)
        if text is None:
            return False
        if opt.type == "button":
            self._enqueue(("cmd", f"setoption name {opt.name}"))
        else:
            self._enqueue(("cmd", f"setoption name {opt.name} value {text}"))
        return True

    def new_game(self) -> None:
        self._drop_queued_searches()
        self._send_stop()
        self._enqueue(("cmd", "ucinewgame"))

    def go(self, fen: str, moves: list[str] | tuple = (), **limits) -> int:
        """Searches `fen` after `moves` (UCI strings) with limits such as
        wtime/btime/winc/binc (ms), movetime, depth, nodes or infinite=True.
        Stops a running search first. Returns the new search's id."""
        if self.state in (EngineState.STOPPED, EngineState.FAILED, EngineState.QUITTING):
            raise RuntimeError(f"{self.name} isn't running ({self.state.value})")
        self._next_id += 1
        sid = self._next_id
        position = f"position fen {fen}" + (" moves " + " ".join(moves) if moves else "")
        self._drop_queued_searches()
        self._send_stop()
        self._enqueue(("search", sid, position, go_command(limits)))
        return sid

    def stop(self) -> None:
        """Ends the running search (its bestmove still arrives, with its id)
        and cancels searches that haven't started."""
        self._drop_queued_searches()
        self._send_stop()

    # --- Internals -------------------------------------------------------------

    def _set_state(self, state: EngineState) -> None:
        if state != self.state:
            self.state = state
            self.state_changed.emit(state)

    def _event(self, text: str) -> None:
        self.log.emit("!", text)

    def _send(self, line: str) -> None:
        if self._proc is None or self._proc.state() != QProcess.ProcessState.Running:
            return
        self.log.emit(">", line)
        self._proc.write((line + "\n").encode())

    def _send_stop(self) -> None:
        if self._active is not None and not self._stopping:
            self._stopping = True
            self._send("stop")

    def _drop_queued_searches(self) -> None:
        self._queue = [item for item in self._queue if item[0] != "search"]

    def _enqueue(self, item: tuple) -> None:
        self._queue.append(item)
        self._pump()

    def _pump(self) -> None:
        """Sends queued items while no search runs; a search ends the run.
        The commands go out before state_changed is emitted, so a listener
        that reacts (even with stop() or go()) sees a consistent engine."""
        while self._queue and self.state in (EngineState.IDLE, EngineState.SEARCHING) and self._active is None:
            item = self._queue.pop(0)
            if item[0] == "cmd":
                self._send(item[1])
            else:
                _, sid, position, go = item
                self._active = sid
                self._stopping = False
                self._send(position)
                self._send(go)
                self._set_state(EngineState.SEARCHING)

    def _fail(self, message: str) -> None:
        if self.state in (EngineState.FAILED, EngineState.QUITTING, EngineState.STOPPED):
            return
        self._timer.stop()
        self._queue.clear()
        self._active = None
        self._event(message)
        self._set_state(EngineState.FAILED)
        if self._proc is not None and self._proc.state() != QProcess.ProcessState.NotRunning:
            self._proc.kill()
        self.failed.emit(f"{self.name}: {message}")

    def _on_started(self) -> None:
        self._send("uci")
        self._waiting_for = "uciok"
        self._timer.start(self._uciok_timeout_ms)

    def _on_timeout(self) -> None:
        self._fail(f"no {self._waiting_for} within {self._timer.interval() / 1000:g} s")

    def _on_process_error(self, error) -> None:
        if error == QProcess.ProcessError.FailedToStart:
            self._fail(f"failed to start ({self._proc.errorString() if self._proc else ''})")
        elif error == QProcess.ProcessError.Crashed and self.state != EngineState.QUITTING:
            self._fail("crashed")

    def _on_finished(self, code: int, status) -> None:
        if self.state not in (EngineState.QUITTING, EngineState.STOPPED, EngineState.FAILED):
            self._fail(f"exited unexpectedly (code {code})")

    def _read_stderr(self) -> None:
        if self._proc is None:
            return
        for line in bytes(self._proc.readAllStandardError()).decode(errors="replace").splitlines():
            if line.strip():
                self.log.emit("<", "stderr: " + line.rstrip())

    def _read_stdout(self) -> None:
        if self._proc is None:
            return
        self._buffer += bytes(self._proc.readAllStandardOutput())
        *lines, self._buffer = self._buffer.split(b"\n")
        for raw in lines:
            line = raw.decode(errors="replace").strip()
            if line:
                self.log.emit("<", line)
                self._handle(line)

    def _handle(self, line: str) -> None:
        word = line.split(maxsplit=1)[0]
        if word == "info":
            if self._active is not None:
                self.info.emit(self._active, parse_info(line))
        elif word == "bestmove":
            if self._active is None:
                self._event("bestmove without a search: ignored")
                return
            sid = self._active
            self._active = None
            self._stopping = False
            best, ponder = parse_bestmove(line)
            self.bestmove.emit(sid, best, ponder)
            self._pump()  # a queued search starts at once (the state stays SEARCHING)
            if self._active is None and self.state == EngineState.SEARCHING:
                self._set_state(EngineState.IDLE)
        elif word == "id":
            parts = line.split(maxsplit=2)
            if len(parts) == 3 and parts[1] == "name":
                self.id_name = parts[2]
            elif len(parts) == 3 and parts[1] == "author":
                self.id_author = parts[2]
        elif word == "option":
            opt = parse_option(line)
            if opt is not None and self.state == EngineState.STARTING:
                self.options.append(opt)
        elif word == "uciok" and self.state == EngineState.STARTING and self._waiting_for == "uciok":
            self._timer.stop()
            self.options_received.emit()
            if self.state != EngineState.STARTING:  # a listener quit the engine (option probing)
                return
            for name, value in self._startup_options.items():
                opt = self.option(name)
                text = opt.normalize(value) if opt else None
                if text is None:
                    self._event(f"ignored option {name}={value} (not offered or invalid)")
                    continue
                self._send(f"setoption name {opt.name}" + ("" if opt.type == "button" else f" value {text}"))
            self._send("isready")
            self._waiting_for = "readyok"
            self._timer.start(self._readyok_timeout_ms)
        elif word == "readyok" and self.state == EngineState.STARTING and self._waiting_for == "readyok":
            self._timer.stop()
            self._waiting_for = ""
            self._set_state(EngineState.IDLE)
            self.ready.emit()
            self._pump()
