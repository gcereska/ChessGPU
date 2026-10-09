"""Engine tournaments: round robins and gauntlets of engine profiles.

Games are played one at a time on the window's game controller (so the
board shows the live game; GPU engines share the GPU). Every pairing plays
each of its openings twice, colors swapped; `rounds` repeats that with the
next openings. Engines stay running between games.

Adjudication, as fishtest's: a win when both engines' reports put the same
side 600 cp or more ahead for their last 3 moves each; a draw from move 34
when both report within 20 cp for their last 8 moves each; optionally a draw
at a move limit. Finished games go to a PGN file at once, and the
tournament's state to a JSON file beside it, from which it can be resumed.

Standings: each engine's score against the field as an Elo difference
(-400 * log10(1/score - 1)) with a 95% error bar from the trinomial
variance of its results, as match runners report it; the cross-table gives
the points of every pairing.
"""

from __future__ import annotations

import json
import math
import pathlib
import random
from dataclasses import asdict, dataclass, field

from PySide6.QtCore import QObject, QTimer, Signal

from .game import GameController, GameRecord, TimeControl
from .pgn import PgnError, read_pgn, write_pgn
from .profiles import EngineProfile
from .rules import WHITE, FenError, Position

KINDS = ("round robin", "gauntlet")
NEXT_GAME_DELAY_MS = 300  # between games: engines finish their last report


@dataclass
class Adjudication:
    resign_cp: int = 600
    resign_moves: int = 3
    draw_cp: int = 20
    draw_moves: int = 8
    draw_from_move: int = 34
    max_moves: int = 0  # a draw after this many moves (0: no limit)
    enabled: bool = True


@dataclass
class TournamentSettings:
    name: str = "Engine tournament"
    engines: list[str] = field(default_factory=list)  # profile names; a gauntlet's first plays all others
    kind: str = "round robin"
    rounds: int = 1  # each round: every pairing plays an opening with both colors
    tc: dict = field(default_factory=lambda: {"base_ms": 60_000, "inc_ms": 600})  # TimeControl fields
    openings: str = ""  # an EPD or PGN file; "" plays from the start position
    opening_plies: int = 8  # moves taken from a PGN file's games
    random_order: bool = True
    seed: int = 1
    adjudication: Adjudication = field(default_factory=Adjudication)

    def time_control(self) -> TimeControl:
        return TimeControl(**self.tc)

    def to_json(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_json(data: dict) -> TournamentSettings:
        data = dict(data)
        adjudication = Adjudication(**data.pop("adjudication", {}))
        known = {k: v for k, v in data.items() if k in TournamentSettings.__dataclass_fields__}
        return TournamentSettings(adjudication=adjudication, **known)


@dataclass(frozen=True)
class Pairing:
    number: int  # 1-based game number
    white: str
    black: str
    opening: int  # index into the openings


@dataclass
class GameResult:
    number: int
    white: str
    black: str
    result: str  # "1-0", "0-1", "1/2-1/2" ("*" for an unfinished game, not counted)
    reason: str = ""
    plies: int = 0


def schedule(engines: list[str], kind: str, rounds: int, openings: int) -> list[Pairing]:
    """Every game, in playing order: round by round, each pairing playing
    its opening with both colors (a new opening per pairing and round)."""
    if kind == "gauntlet":
        pairs = [(engines[0], e) for e in engines[1:]]
    else:
        pairs = [(a, b) for i, a in enumerate(engines) for b in engines[i + 1:]]
    out = []
    opening = 0
    for _ in range(max(1, rounds)):
        for a, b in pairs:
            o = opening % max(1, openings)
            out.append(Pairing(len(out) + 1, a, b, o))
            out.append(Pairing(len(out) + 1, b, a, o))
            opening += 1
    return out


def load_openings(path: str, plies: int = 8, random_order: bool = False, seed: int = 1) -> list[Position]:
    """Opening positions from an EPD file (a FEN per line, before any ";"),
    or from a PGN file (each game's position after its first `plies` moves).
    No file: the start position. Raises ValueError for unreadable files."""
    if not path:
        return [Position.start()]
    text = pathlib.Path(path).read_text(encoding="utf-8", errors="replace")
    out: list[Position] = []
    if path.lower().endswith(".pgn"):
        try:
            games = read_pgn(text)
        except PgnError as e:
            raise ValueError(f"{path}: {e}") from None
        for g in games:
            pos = g.start
            for m in g.moves[:max(0, plies)]:
                pos = pos.push(m)
            out.append(pos)
    else:
        for n, line in enumerate(text.splitlines(), start=1):
            line = line.split(";")[0].strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            fen = " ".join(fields[:6]) if len(fields) >= 6 and fields[4].isdigit() else " ".join(fields[:4]) + " 0 1"
            try:
                out.append(Position.from_fen(fen))
            except FenError as e:
                raise ValueError(f"{path}, line {n}: {e}") from None
    if not out:
        raise ValueError(f"{path}: no openings")
    if random_order:
        random.Random(seed).shuffle(out)
    return out


# --- Results -------------------------------------------------------------------------------------


def elo_of(score: float) -> float:
    """The Elo difference expected to score `score` (0..1, exclusive)."""
    return -400.0 * math.log10(1.0 / score - 1.0)


def elo_with_error(wins: int, draws: int, losses: int) -> tuple[float | None, float | None]:
    """(Elo difference, 95% error) from results; None when undefined (no
    games, or all won or all lost)."""
    n = wins + draws + losses
    if n == 0:
        return None, None
    s = (wins + draws / 2) / n
    if s <= 0 or s >= 1:
        return None, None
    var = (wins * (1 - s) ** 2 + draws * (0.5 - s) ** 2 + losses * s ** 2) / n
    margin = 1.959964 * math.sqrt(var / n)
    lo, hi = max(1e-6, s - margin), min(1 - 1e-6, s + margin)
    return elo_of(s), (elo_of(hi) - elo_of(lo)) / 2


def los(wins: int, losses: int) -> float:
    """Likelihood of superiority (draws carry no information)."""
    if wins + losses == 0:
        return 0.5
    return 0.5 + 0.5 * math.erf((wins - losses) / math.sqrt(2.0 * (wins + losses)))


@dataclass
class Standing:
    engine: str
    games: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0

    @property
    def points(self) -> float:
        return self.wins + self.draws / 2

    @property
    def score(self) -> float:
        return self.points / self.games if self.games else 0.0

    def elo(self) -> tuple[float | None, float | None]:
        return elo_with_error(self.wins, self.draws, self.losses)


def counted(results: list[GameResult]) -> list[GameResult]:
    return [r for r in results if r.result in ("1-0", "0-1", "1/2-1/2")]


def standings(engines: list[str], results: list[GameResult]) -> list[Standing]:
    """Every engine's results, best first (points, then score)."""
    table = {e: Standing(e) for e in engines}
    for r in counted(results):
        for engine, won in ((r.white, r.result == "1-0"), (r.black, r.result == "0-1")):
            s = table.setdefault(engine, Standing(engine))
            s.games += 1
            if r.result == "1/2-1/2":
                s.draws += 1
            elif won:
                s.wins += 1
            else:
                s.losses += 1
    return sorted(table.values(), key=lambda s: (-s.points, -s.score, engines.index(s.engine)
                                                  if s.engine in engines else 0))


def crosstable(engines: list[str], results: list[GameResult]) -> dict[tuple[str, str], tuple[float, int]]:
    """(row engine, column engine) -> (row's points against column, games)."""
    out: dict[tuple[str, str], tuple[float, int]] = {}
    for r in counted(results):
        white_points = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}[r.result]
        for a, b, pts in ((r.white, r.black, white_points), (r.black, r.white, 1.0 - white_points)):
            p, g = out.get((a, b), (0.0, 0))
            out[(a, b)] = (p + pts, g + 1)
    return out


class Adjudicator:
    """Decides, after each move, whether to end a game by the engines'
    agreement (see the module's docstring)."""

    def __init__(self, rules: Adjudication):
        self.rules = rules

    def verdict(self, record: GameRecord) -> tuple[str, str] | None:
        r = self.rules
        if not r.enabled or not record.moves:
            return None
        if r.max_moves and record.position.fullmove > r.max_moves:
            return "1/2-1/2", "Draw by adjudication (move limit)"
        views = []  # White's view of each move's report, latest first
        for i in range(len(record.moves) - 1, -1, -1):
            m = record.moves[i]
            if m.eval is None or m.eval.score is None:
                break
            cp = m.eval.score.bar_cp()
            views.append(cp if record.positions[i].turn == WHITE else -cp)
        n = 2 * r.resign_moves
        if r.resign_moves and len(views) >= n:
            last = views[:n]
            if all(v >= r.resign_cp for v in last):
                return "1-0", "White wins by adjudication"
            if all(v <= -r.resign_cp for v in last):
                return "0-1", "Black wins by adjudication"
        n = 2 * r.draw_moves
        if r.draw_moves and len(views) >= n and record.position.fullmove >= r.draw_from_move:
            if all(abs(v) <= r.draw_cp for v in views[:n]):
                return "1/2-1/2", "Draw by adjudication"
        return None


# --- Running a tournament --------------------------------------------------------------------------


class TournamentController(QObject):
    """Plays a tournament's games on a game controller, one at a time."""

    game_started = Signal(object)  # Pairing
    game_finished = Signal(object)  # GameResult
    changed = Signal()  # results, state
    finished = Signal(bool)  # True: every game played; False: stopped
    message = Signal(str)

    IDLE, RUNNING, PAUSED = "idle", "running", "paused"

    def __init__(self, game: GameController, profiles, parent: QObject | None = None):
        super().__init__(parent)
        self.game = game
        self.profiles = profiles  # callable: name -> EngineProfile | None
        self.settings: TournamentSettings | None = None
        self.pairings: list[Pairing] = []
        self.results: list[GameResult] = []
        self.openings: list[Position] = []
        self.pgn_path: pathlib.Path | None = None
        self.state = self.IDLE
        self.current: Pairing | None = None
        self._pause_requested = False
        self.adjudicator = Adjudicator(Adjudication())
        game.game_over.connect(self._on_game_over)
        game.changed.connect(self._check_adjudication)

    @property
    def active(self) -> bool:
        """A tournament is under way (running or paused)."""
        return self.state != self.IDLE

    @property
    def pausing(self) -> bool:
        """A pause is asked for: it comes once the current game ends."""
        return self._pause_requested and self.state == self.RUNNING

    def start(self, settings: TournamentSettings, pgn_path: str, results: list[GameResult] | None = None) -> None:
        """Starts (or, with earlier results, resumes) a tournament; its games
        are appended to `pgn_path`, its state saved beside it (.json)."""
        if self.active:
            raise RuntimeError("a tournament is under way")
        missing = [name for name in settings.engines if self.profiles(name) is None]
        if len(settings.engines) < 2 or missing:
            raise ValueError("a tournament needs two engines or more" if not missing else
                             f"no engine profile named {', '.join(missing)}")
        self.openings = load_openings(settings.openings, settings.opening_plies, settings.random_order,
                                      settings.seed)
        self.settings = settings
        self.pairings = schedule(settings.engines, settings.kind, settings.rounds, len(self.openings))
        self.results = list(results or [])
        self.pgn_path = pathlib.Path(pgn_path)
        self.adjudicator = Adjudicator(settings.adjudication)
        self.state = self.RUNNING
        self._pause_requested = False
        self.game.keep_engines(True)
        self.save_state()
        self.changed.emit()
        self._next()

    def pause(self) -> None:
        """Lets the game being played finish, then waits."""
        if self.state == self.RUNNING:
            self._pause_requested = True
            if self.current is None:
                self.state = self.PAUSED
            self.changed.emit()

    def resume(self) -> None:
        if not self.active:
            return
        self._pause_requested = False
        if self.state == self.PAUSED:
            self.state = self.RUNNING
            self.changed.emit()
            self._next()

    def stop(self) -> None:
        """Ends the tournament; the game being played is aborted (not counted)."""
        if not self.active:
            return
        self.current = None  # (its abort isn't a result)
        self.state = self.IDLE
        if self.game.status in (self.game.RUNNING, self.game.PREPARING):
            self.game.abort()
        self.game.keep_engines(False)
        self.save_state()
        self.changed.emit()
        self.finished.emit(False)

    def remaining(self) -> list[Pairing]:
        done = {r.number for r in counted(self.results)}
        return [p for p in self.pairings if p.number not in done]

    def standings(self) -> list[Standing]:
        return standings(self.settings.engines if self.settings else [], self.results)

    def state_path(self) -> pathlib.Path | None:
        return self.pgn_path.with_suffix(".json") if self.pgn_path else None

    def save_state(self) -> None:
        path = self.state_path()
        if path is None or self.settings is None:
            return
        data = {"settings": self.settings.to_json(), "pgn": str(self.pgn_path),
                "results": [asdict(r) for r in counted(self.results)]}
        try:
            path.write_text(json.dumps(data, indent=1), encoding="utf-8")
        except OSError as e:
            self.message.emit(f"Couldn't save the tournament state: {e}")

    @staticmethod
    def load_state(path: str) -> tuple[TournamentSettings, str, list[GameResult]]:
        """A saved tournament: (settings, PGN path, results). Raises ValueError."""
        try:
            data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
            return (TournamentSettings.from_json(data["settings"]), data["pgn"],
                    [GameResult(**r) for r in data.get("results", [])])
        except (OSError, ValueError, KeyError, TypeError) as e:
            raise ValueError(f"{path}: not a tournament ({e})") from None

    def _next(self) -> None:
        if self.state != self.RUNNING:  # (a pause asked for between games took effect at once)
            return
        todo = self.remaining()
        if not todo:
            self.state = self.IDLE
            self.current = None
            self.game.keep_engines(False)
            self.save_state()
            self.changed.emit()
            self.finished.emit(True)
            return
        p = todo[0]
        self.current = p
        s = self.settings
        start = self.openings[p.opening % len(self.openings)]
        self.game.book, self.game.book_plies = None, 0  # (the openings come from the tournament's file)
        self.game.new_game(self.profiles(p.white), self.profiles(p.black), s.time_control(), start, event=s.name)
        self.game.record.tags["Round"] = str(p.number)
        self.game_started.emit(p)
        self.changed.emit()

    def _check_adjudication(self) -> None:
        if self.state == self.IDLE or self.current is None or self.game.status != self.game.RUNNING:
            return
        verdict = self.adjudicator.verdict(self.game.record)
        if verdict is not None:
            QTimer.singleShot(0, lambda v=verdict, p=self.current: self._adjudicate(p, v))

    def _adjudicate(self, pairing: Pairing, verdict: tuple[str, str]) -> None:
        if self.current is pairing and self.game.status == self.game.RUNNING:
            self.game.adjudicate(*verdict)

    def _on_game_over(self, result: str, reason: str) -> None:
        p = self.current
        if p is None or self.state == self.IDLE:
            return
        self.current = None
        record = self.game.record
        r = GameResult(p.number, p.white, p.black, result, reason, len(record.moves))
        if result in ("1-0", "0-1", "1/2-1/2"):
            self.results.append(r)
            self._append_pgn(record)
        else:
            self.message.emit(f"Game {p.number} ended without a result ({reason}); it will be played again.")
        self.save_state()
        self.game_finished.emit(r)
        self.changed.emit()
        if self._pause_requested:
            self.state = self.PAUSED
            self.changed.emit()
        elif result == "*":
            self.stop()  # (an engine that fails to start would fail again)
        else:
            QTimer.singleShot(NEXT_GAME_DELAY_MS, self._next)

    def _append_pgn(self, record: GameRecord) -> None:
        try:
            with open(self.pgn_path, "a", encoding="utf-8") as f:
                f.write(write_pgn(record.to_pgn()) + "\n")
        except OSError as e:
            self.message.emit(f"Couldn't write {self.pgn_path}: {e}")
