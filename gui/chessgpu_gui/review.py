"""Game review: an engine evaluates every position of the main line; each
move is then judged by how much of the mover's expected score it gave up.

Expected score: the logistic of the engine's centipawns used across the GUI
(panels.win_probability: 1 / (1 + 10^(-cp/400)), mates 1 or 0), seen by the
side that moved. A move's loss is its expected score before the move minus
after it (none if the engine's own best move was played); 7% or more is an
inaccuracy (?!), 14% a mistake (?), 25% a blunder (??). A move's accuracy is
100 * exp(-4 * loss), a side's accuracy their moves' mean; the average
centipawn loss counts scores capped at +-1000 (mates included).

The review can annotate the game: NAGs on judged moves, a comment naming
the engine's choice, and its line as a variation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from PySide6.QtCore import QObject, Signal

from .game import GameRecord, Node
from .panels import win_probability
from .profiles import EngineProfile
from .rules import BLACK, WHITE, IllegalMoveError
from .score import PositionEval, Score
from .uci import EngineState, Info, UciEngine

INACCURACY, MISTAKE, BLUNDER = 0.07, 0.14, 0.25
KINDS = ("inaccuracy", "mistake", "blunder")
KIND_NAGS = {"inaccuracy": 6, "mistake": 2, "blunder": 4}
KIND_TITLES = {"inaccuracy": "Inaccuracy", "mistake": "Mistake", "blunder": "Blunder"}
BEST_LINE_PLIES = 8  # of the engine's line, added as a variation when annotating


def expected_score(score: Score, color: int) -> float:
    """`color`'s expected score (0..1) from a White's-view score."""
    p = win_probability(score)
    return p if color == WHITE else 1.0 - p


def capped_cp(score: Score, color: int) -> int:
    """`color`'s view of a White's-view score in centipawns, capped at
    +-1000 (score.DECIDED_CP; mates at the cap)."""
    cp = score.bar_cp()
    return cp if color == WHITE else -cp


def kind_of(loss: float) -> str | None:
    if loss >= BLUNDER:
        return "blunder"
    if loss >= MISTAKE:
        return "mistake"
    if loss >= INACCURACY:
        return "inaccuracy"
    return None


def accuracy_of(loss: float) -> float:
    return 100.0 * math.exp(-4.0 * loss)


@dataclass
class Judgement:
    """What the review makes of one main-line move."""

    node: Node
    mover: int
    loss: float  # of expected score (0..1)
    cp_loss: int  # capped centipawns
    kind: str | None  # "inaccuracy", "mistake", "blunder" or None
    best: str | None  # the engine's move in the position before (UCI), if another one
    best_line: tuple[str, ...]  # its line from there

    @property
    def accuracy(self) -> float:
        return accuracy_of(self.loss)


@dataclass
class SideSummary:
    moves: int = 0
    accuracy: float = 0.0  # mean of the judged moves', 0 if none
    acpl: float = 0.0  # average centipawn loss
    inaccuracies: int = 0
    mistakes: int = 0
    blunders: int = 0


def judge(record: GameRecord) -> list[Judgement]:
    """A judgement for every main-line move whose position before and after
    were both reviewed."""
    out = []
    for node in record.moves:
        before, after = node.parent.review, node.review
        if before is None or after is None:
            continue
        mover = node.parent.position.turn
        best = before.pv[0] if before.pv else None
        if best == node.move.uci():
            loss, cp_loss, best, line = 0.0, 0, None, ()
        else:
            loss = max(0.0, expected_score(before.score, mover) - expected_score(after.score, mover))
            cp_loss = max(0, capped_cp(before.score, mover) - capped_cp(after.score, mover))
            line = before.pv
        out.append(Judgement(node, mover, loss, cp_loss, kind_of(loss), best, tuple(line)))
    return out


def summarize(judgements: list[Judgement]) -> dict[int, SideSummary]:
    sides = {WHITE: SideSummary(), BLACK: SideSummary()}
    for j in judgements:
        s = sides[j.mover]
        s.moves += 1
        s.accuracy += j.accuracy
        s.acpl += j.cp_loss
        s.inaccuracies += j.kind == "inaccuracy"
        s.mistakes += j.kind == "mistake"
        s.blunders += j.kind == "blunder"
    for s in sides.values():
        if s.moves:
            s.accuracy /= s.moves
            s.acpl /= s.moves
    return sides


def annotate(record: GameRecord, judgements: list[Judgement], variations: bool = True) -> int:
    """Marks the inaccuracies, mistakes and blunders: a NAG (replacing other
    move NAGs), a comment naming the engine's move, and its line as a
    variation. Running it again changes nothing. Returns the moves marked."""
    marked = 0
    for j in judgements:
        if j.kind is None:
            continue
        node = j.node
        marked += 1
        node.nags = [n for n in node.nags if n > 7] + [KIND_NAGS[j.kind]]
        before = node.parent.position
        text = KIND_TITLES[j.kind] + "."
        if j.best:
            try:
                text += f" {before.san(before.parse_uci(j.best))} was best."
            except IllegalMoveError:
                pass
        if text not in node.comment:
            node.comment = f"{text} {node.comment}".strip()
        if variations and j.best_line:
            moves, pos = [], before
            for uci in j.best_line[:BEST_LINE_PLIES]:
                try:
                    m = pos.parse_uci(uci)
                except IllegalMoveError:
                    break
                moves.append(m)
                pos = pos.push(m)
            if moves:
                record.add_line(node.parent, moves)
    return marked


class ReviewController(QObject):
    """Evaluates the main line's positions one after the other with an
    engine (each for `movetime_ms`, or to `depth`): sets node.review."""

    progress = Signal(int, int)  # positions done, of
    finished = Signal(bool)  # True: all done; False: cancelled or failed
    message = Signal(str)
    engine_log = Signal(str, str, str)

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self.engine: UciEngine | None = None
        self.record: GameRecord | None = None
        self.nodes: list[Node] = []
        self.done = 0
        self.movetime_ms: int | None = 1000
        self.depth: int | None = None
        self._search: int | None = None
        self._last: Info | None = None

    @property
    def running(self) -> bool:
        return self.record is not None

    def start(self, record: GameRecord, profile: EngineProfile, movetime_ms: int | None = 1000,
              depth: int | None = None) -> None:
        self.cancel()
        self.record = record
        self.nodes = [record.root] + list(record.moves)
        self.done = 0
        self.movetime_ms, self.depth = movetime_ms, depth
        e = UciEngine(f"review engine ({profile.name})", self)
        e.info.connect(self._on_info)
        e.bestmove.connect(self._on_bestmove)
        e.failed.connect(self._on_failed)
        e.ready.connect(self._next)
        e.log.connect(lambda d, line, n=e.name: self.engine_log.emit(n, d, line))
        self.engine = e
        options = {k: v for k, v in profile.options.items() if k.lower() != "multipv"}
        e.start(profile.path, profile.args, options)
        self.progress.emit(0, len(self.nodes))

    def cancel(self) -> None:
        """Stops; what was reviewed stays."""
        if self.record is None:
            return
        self._stop_engine()
        self.finished.emit(False)

    def _stop_engine(self) -> None:
        self.record = None
        self._search = None
        if self.engine is not None:
            self.engine.quit()
            self.engine = None

    def _next(self) -> None:
        """Reviews the next position (positions without moves need no engine)."""
        while self.record is not None and self.done < len(self.nodes):
            node = self.nodes[self.done]
            pos = node.position
            if pos.legal_moves():
                self._last = None
                limits = {"depth": self.depth} if self.depth else {"movetime": self.movetime_ms or 1000}
                self._search = -1
                self._search = self.engine.go(self.record.start.fen(), node.uci_moves(), **limits)
                return
            if pos.in_check():  # checkmate: the side to move is mated (White's view)
                score = Score(mate=0) if pos.turn == WHITE else Score(mate=0).negated()
            else:  # stalemate
                score = Score(cp=0)
            node.review = PositionEval(score, 0)
            self.done += 1
            self.progress.emit(self.done, len(self.nodes))
        if self.record is not None:
            self._stop_engine()
            self.finished.emit(True)

    def _on_info(self, sid: int, info: Info) -> None:
        if sid == self._search and info.score is not None and info.bound is None and (info.multipv or 1) == 1:
            self._last = info

    def _on_bestmove(self, sid: int, best, _ponder) -> None:
        if sid != self._search or self.record is None:
            return
        self._search = None
        node = self.nodes[self.done]
        info = self._last
        if info is not None:
            turn = node.position.turn
            score = info.score if turn == WHITE else info.score.negated()
            node.review = PositionEval(score, info.depth, tuple(info.pv))
        self.done += 1
        self.progress.emit(self.done, len(self.nodes))
        self._next()

    def _on_failed(self, message: str) -> None:
        self.message.emit(message)
        if self.record is not None:
            self.record = None
            self._search = None
            self.engine = None
            self.finished.emit(False)
