"""Game databases: PGN files opened as lists of games, searched by tags and
by position, with an opening explorer over them (which also serves engines
as an opening book).

Opening a file only scans it (each game's tags and text), so large files
open fast; a game's moves are read when it's shown. The position index
records, for every position each game reaches in its first `plies` moves,
the moves played from it with their results, and which games reached it.
It is built in worker processes, a chunk of games at a time (the rules
module is pure Python: about 15k moves a second per process).
"""

from __future__ import annotations

import concurrent.futures
import os
import pathlib
import random
import re
import threading
from dataclasses import dataclass, field

from PySide6.QtCore import QObject, Signal

from .game import GameRecord
from .pgn import _TOKEN_RE, PgnError, read_pgn
from .rules import FenError, IllegalMoveError, Move, Position

INDEX_PLIES = 40  # plies (half-moves: 20 moves) of each game the position index covers
CHUNK = 400  # games per worker task
RESULT_SLOTS = {"1-0": 0, "1/2-1/2": 1, "0-1": 2}  # stats: [White wins, draws, Black wins, unfinished]

_TAG_RE = re.compile(r'\[\s*([A-Za-z0-9_]+)\s+"((?:[^"\\]|\\.)*)"\s*\]')


def explorer_key(pos: Position) -> tuple:
    """What makes two positions the same for the explorer (any en passant
    square counts, unlike repetitions)."""
    return pos.board, pos.turn, pos.castling, pos.ep


@dataclass
class DbGame:
    number: int  # 1-based, across the database's files
    tags: dict[str, str]
    text: str  # the game's PGN text

    def tag(self, name: str) -> str:
        return self.tags.get(name, "")

    @property
    def result(self) -> str:
        r = self.tags.get("Result", "*")
        return r if r in ("1-0", "0-1", "1/2-1/2") else "*"

    def record(self) -> GameRecord:
        """The game, read in full. Raises PgnError."""
        games = read_pgn(self.text)
        if not games:
            raise PgnError(f"game {self.number}: no moves")
        return GameRecord.from_pgn(games[0])


def scan_pgn(text: str, first_number: int = 1) -> list[DbGame]:
    """The games of a PGN text: tags and text each, moves unread. A game
    starts at a tag line after the previous game's moves."""
    games: list[DbGame] = []
    lines: list[str] = []
    tags: dict[str, str] = {}
    moves_seen = False

    def flush():
        if tags or any(l.strip() for l in lines):
            games.append(DbGame(first_number + len(games), dict(tags), "\n".join(lines)))

    for line in text.splitlines():
        if line.startswith("%"):
            continue
        stripped = line.lstrip()
        if stripped.startswith("["):
            if moves_seen:
                flush()
                lines, tags, moves_seen = [], {}, False
            for m in _TAG_RE.finditer(stripped):
                tags[m.group(1)] = re.sub(r"\\(.)", r"\1", m.group(2))
        elif stripped:
            moves_seen = True
        lines.append(line)
    flush()
    return games


def mainline(text: str, limit: int) -> tuple[Position, list[Move]]:
    """A game's start and its first `limit` main-line moves, read quickly
    (comments, NAGs and variations skipped). Raises PgnError."""
    tags = dict((m.group(1), m.group(2)) for m in _TAG_RE.finditer(text))
    try:
        pos = Position.from_fen(tags["FEN"]) if "FEN" in tags else Position.start()
    except FenError as e:
        raise PgnError(str(e)) from None
    start = pos
    moves: list[Move] = []
    depth = 0
    body = _TAG_RE.sub(" ", text)
    for m in _TOKEN_RE.finditer(body):
        kind = m.lastgroup
        if kind == "open":
            depth += 1
        elif kind == "close":
            depth = max(0, depth - 1)
        elif kind == "result" and depth == 0:
            break
        elif kind == "move" and depth == 0:
            if len(moves) >= limit:
                break
            san = m.group("move").rstrip("!?")
            try:
                move = pos.parse_san(san)
            except IllegalMoveError as e:
                raise PgnError(str(e)) from None
            moves.append(move)
            pos = pos.push(move)
    return start, moves


@dataclass
class PositionIndex:
    """Moves played from each position (with results) and the games that
    reached it, over the first `plies` moves of every game."""

    plies: int = INDEX_PLIES
    moves: dict[tuple, dict[str, list[int]]] = field(default_factory=dict)  # key -> uci -> [W, D, B, *]
    games: dict[tuple, list[int]] = field(default_factory=dict)  # key -> game numbers
    indexed: int = 0  # games read
    failed: int = 0  # games that couldn't be read

    def merge(self, other: PositionIndex) -> None:
        for key, moves in other.moves.items():
            mine = self.moves.setdefault(key, {})
            for uci, stats in moves.items():
                total = mine.setdefault(uci, [0, 0, 0, 0])
                for i, v in enumerate(stats):
                    total[i] += v
        for key, numbers in other.games.items():
            self.games.setdefault(key, []).extend(numbers)
        self.indexed += other.indexed
        self.failed += other.failed

    def stats(self, pos: Position) -> list[tuple[Move, list[int]]]:
        """The moves played from `pos`, most played first: (move, [W, D, B, *])."""
        found = self.moves.get(explorer_key(pos), {})
        out = []
        for uci, stats in found.items():
            try:
                out.append((pos.parse_uci(uci), stats))
            except IllegalMoveError:
                continue
        return sorted(out, key=lambda x: (-sum(x[1]), x[0].uci()))

    def games_at(self, pos: Position) -> list[int]:
        return sorted(set(self.games.get(explorer_key(pos), [])))

    def book_move(self, pos: Position, rng: random.Random, min_games: int = 1) -> Move | None:
        """A move for an engine from the book: chosen at random, in
        proportion to how often it was played."""
        choices = [(m, sum(s)) for m, s in self.stats(pos) if sum(s) >= min_games]
        if not choices:
            return None
        total = sum(n for _, n in choices)
        pick = rng.uniform(0, total)
        for move, n in choices:
            pick -= n
            if pick <= 0:
                return move
        return choices[-1][0]


def index_games(items: list[tuple[int, str, str]], plies: int) -> PositionIndex:
    """The position index of games given as (number, result, PGN text) (a
    worker process's task)."""
    index = PositionIndex(plies)
    for number, result, text in items:
        try:
            start, moves = mainline(text, plies)
        except PgnError:
            index.failed += 1
            continue
        slot = RESULT_SLOTS.get(result, 3)
        pos = start
        seen = set()
        for move in moves:
            key = explorer_key(pos)
            if key not in seen:
                seen.add(key)
                index.games.setdefault(key, []).append(number)
            index.moves.setdefault(key, {}).setdefault(move.uci(), [0, 0, 0, 0])[slot] += 1
            pos = pos.push(move)
        key = explorer_key(pos)
        if key not in seen:
            index.games.setdefault(key, []).append(number)
        index.indexed += 1
    return index


class GameDatabase(QObject):
    """Games from PGN files, their search, and their position index
    (built in the background: see build_index)."""

    changed = Signal()  # games added or removed
    index_progress = Signal(int, int)  # games indexed, of
    index_ready = Signal()
    index_failed = Signal(str)  # why (the index stays as it was: none)
    index_message = Signal(str)  # e.g. worker processes couldn't run: indexed here instead

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self.games: list[DbGame] = []
        self.paths: list[str] = []
        self.index: PositionIndex | None = None
        self._job = 0  # the index build that counts (a newer one makes an older one's result stale)
        self._building = False

    @property
    def name(self) -> str:
        if not self.paths:
            return ""
        first = pathlib.Path(self.paths[0]).name
        return first if len(self.paths) == 1 else f"{first} and {len(self.paths) - 1} more"

    @property
    def building(self) -> bool:
        return self._building

    def add_file(self, path: str) -> int:
        """Adds a PGN file's games; returns how many. Raises OSError."""
        text = pathlib.Path(path).read_text(encoding="utf-8", errors="replace")
        new = scan_pgn(text, len(self.games) + 1)
        self.games += new
        self.paths.append(str(path))
        self.index = None
        self._job += 1
        self.changed.emit()
        return len(new)

    def clear(self) -> None:
        self.games, self.paths, self.index = [], [], None
        self._job += 1
        self._building = False
        self.changed.emit()

    def game(self, number: int) -> DbGame | None:
        return self.games[number - 1] if 1 <= number <= len(self.games) else None

    def search(self, text: str = "", result: str = "", position: Position | None = None) -> list[DbGame]:
        """Games whose players, event, site or opening contain `text`
        (case-insensitive), with `result` if given, reaching `position`
        (within the indexed moves; needs the index)."""
        needle = text.strip().lower()
        allowed = None
        if position is not None:
            allowed = set(self.index.games_at(position)) if self.index else set()
        out = []
        for g in self.games:
            if allowed is not None and g.number not in allowed:
                continue
            if result and g.result != result:
                continue
            if needle and not any(needle in g.tags.get(k, "").lower()
                                  for k in ("White", "Black", "Event", "Site", "Opening", "ECO")):
                continue
            out.append(g)
        return out

    def build_index(self, plies: int = INDEX_PLIES, workers: int | None = None, wait: bool = False,
                    chunk: int = CHUNK) -> None:
        """Builds the position index in worker processes (`workers` 0: in
        this process), `chunk` games per task, in a background thread
        unless `wait`."""
        self._job += 1
        job = self._job
        items = [(g.number, g.result, g.text) for g in self.games]
        self._building = True

        def in_process(chunks) -> PositionIndex:
            index, done = PositionIndex(plies), 0
            for part in chunks:
                if job != self._job:
                    return index
                index.merge(index_games(part, plies))
                done += len(part)
                self.index_progress.emit(done, len(items))
            return index

        def in_workers(chunks) -> PositionIndex | None:
            index, done = PositionIndex(plies), 0
            count = workers or max(1, min(8, (os.cpu_count() or 2) - 1))
            with concurrent.futures.ProcessPoolExecutor(max_workers=count) as pool:
                futures = {pool.submit(index_games, part, plies): len(part) for part in chunks}
                for f in concurrent.futures.as_completed(futures):
                    if job != self._job:
                        for other in futures:
                            other.cancel()
                        return None
                    index.merge(f.result())
                    done += futures[f]
                    self.index_progress.emit(done, len(items))
            return index

        def run():
            chunks = [items[i:i + chunk] for i in range(0, len(items), max(1, chunk))]
            try:
                if workers == 0 or len(chunks) <= 1:
                    index = in_process(chunks)
                else:
                    try:
                        index = in_workers(chunks)
                    except (concurrent.futures.BrokenExecutor, OSError, RuntimeError) as e:
                        self.index_message.emit(f"Worker processes failed ({e or type(e).__name__}): "
                                                "indexing in this process instead.")
                        index = in_process(chunks)
            except Exception as e:  # noqa: BLE001 (a bug must not leave "indexing..." forever)
                if job == self._job:
                    self._building = False
                    self.index_failed.emit(f"{type(e).__name__}: {e}")
                return
            if job == self._job and index is not None:
                self.index = index
                self._building = False
                self.index_ready.emit()

        if wait:
            run()
        else:
            threading.Thread(target=run, daemon=True, name="position index").start()
