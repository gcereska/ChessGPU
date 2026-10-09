"""Games: the record of a game (a tree of moves: the main line and
variations), clocks, and the controllers that run play (a human or an engine
on each side) and analysis.

The game controller is the arbiter: it checks every engine move, ends games
by the rules (rules.rules_ending), on time, by illegal moves, failed engines
or resignation, and starts the clock only once every engine of the game has
finished its handshake (engine start-up, e.g. GPU set-up, costs no clock
time).
"""

from __future__ import annotations

import random
import re
import time
from dataclasses import dataclass
from datetime import date

from PySide6.QtCore import QObject, QTimer, Signal

from .pgn import PgnGame, join_commands, split_commands
from .profiles import EngineProfile
from .rules import (BLACK, WHITE, Ending, IllegalMoveError, Move, Position, other, parse_square, rules_ending,
                    square_name)
from .score import MoveEval, PositionEval, parse_eval_command, parse_eval_comment
from .uci import EngineState, Info, UciEngine

COLOR_NAMES = ("White", "Black")
DEFAULT_ENGINE_MOVETIME_MS = 1000  # an engine with no limit at all thinks this long
BOOK_DELAY_MS = 120  # an engine's book move comes after this (so the board can show the move before)


def win_for(color: int) -> str:
    return "1-0" if color == WHITE else "0-1"


def ending_result(pos: Position, ending: Ending) -> tuple[str, str]:
    """(PGN result, reason) for a game the rules ended in `pos`."""
    if ending == Ending.CHECKMATE:
        winner = other(pos.turn)
        return win_for(winner), f"{COLOR_NAMES[winner]} mates"
    reasons = {Ending.STALEMATE: "Draw by stalemate",
               Ending.INSUFFICIENT_MATERIAL: "Draw by insufficient material",
               Ending.FIFTY_MOVES: "Draw by the fifty-move rule",
               Ending.THREEFOLD_REPETITION: "Draw by threefold repetition"}
    return "1/2-1/2", reasons[ending]


@dataclass(frozen=True)
class TimeControl:
    """A clock (base + increment, in ms), or per-move limits for engines."""

    base_ms: int | None = None
    inc_ms: int = 0
    movetime_ms: int | None = None
    depth: int | None = None
    nodes: int | None = None

    @property
    def clocked(self) -> bool:
        return self.base_ms is not None

    def text(self) -> str:
        if self.clocked:
            return f"{self.base_ms / 1000:g}+{self.inc_ms / 1000:g}"
        parts = []
        if self.movetime_ms:
            parts.append(f"{self.movetime_ms / 1000:g} s per move")
        if self.depth:
            parts.append(f"depth {self.depth}")
        if self.nodes:
            parts.append(f"{self.nodes} nodes")
        return ", ".join(parts) or "untimed"

    def pgn_tag(self) -> str:
        return f"{self.base_ms / 1000:g}+{self.inc_ms / 1000:g}" if self.clocked else "-"

    def engine_limits(self, white_ms: float, black_ms: float) -> dict:
        if self.clocked:
            return {"wtime": max(0, round(white_ms)), "btime": max(0, round(black_ms)), "winc": self.inc_ms,
                    "binc": self.inc_ms}
        limits = {k: v for k, v in (("movetime", self.movetime_ms), ("depth", self.depth), ("nodes", self.nodes)) if v}
        return limits or {"movetime": DEFAULT_ENGINE_MOVETIME_MS}


class GameClock:
    """Both players' remaining time; runs for one side at a time. Also
    measures every move's thinking time in untimed games."""

    def __init__(self, tc: TimeControl, now=time.monotonic):
        self.tc = tc
        self._now = now
        self.remaining = [float(tc.base_ms or 0), float(tc.base_ms or 0)]
        self.running: int | None = None
        self._started = 0.0

    def start(self, color: int) -> None:
        self.running = color
        self._started = self._now()

    def elapsed_ms(self) -> float:
        return (self._now() - self._started) * 1000 if self.running is not None else 0.0

    def stop(self) -> float:
        """Stops the running side's clock; returns the time it ran (ms)."""
        if self.running is None:
            return 0.0
        elapsed = self.elapsed_ms()
        self.remaining[self.running] -= elapsed
        self.running = None
        return elapsed

    def left(self, color: int) -> float:
        return self.remaining[color] - (self.elapsed_ms() if color == self.running else 0.0)

    def add_increment(self, color: int) -> None:
        self.remaining[color] += self.tc.inc_ms


SHAPE_LETTERS = {"green": "G", "red": "R", "blue": "B", "yellow": "Y"}
_LETTER_SHAPES = {v: k for k, v in SHAPE_LETTERS.items()}
_SHAPE_RE = re.compile(r"^(?P<color>[GRBY])(?P<a>[a-h][1-8])(?P<b>[a-h][1-8])?$")
_CLOCK_RE = re.compile(r"^(?P<h>\d+):(?P<m>\d{1,2}):(?P<s>\d{1,2}(?:\.\d+)?)$")
_NOTE_RE = re.compile(r"^[\w.-]+=\S*$")  # "tl=1.974s", "nps=4229086": match runners' notes
_REASON_RE = re.compile(r"^(?:(?:White|Black)(?:'s\s|\s+(?:mates|resigns|loses|wins|makes|returned|disconnects|"
                        r"terminates|forfeits|abandons|illegal)\b)|Draw\b|Game (?:aborted|not started)\b)")


def format_clock_command(ms: float) -> str:
    """A [%clk] value: h:mm:ss, with tenths when there are any."""
    tenths = int(round(max(0.0, ms) / 100))
    s, t = divmod(tenths, 10)
    return f"{s // 3600}:{s // 60 % 60:02d}:{s % 60:02d}" + (f".{t}" if t else "")


def parse_clock_command(text: str) -> float | None:
    """Milliseconds from a [%clk] value; None if it isn't one."""
    m = _CLOCK_RE.match(text.strip())
    if not m:
        return None
    return (int(m.group("h")) * 3600 + int(m.group("m")) * 60 + float(m.group("s"))) * 1000


def shapes_commands(arrows: dict[tuple[int, int], str], circles: dict[int, str]) -> dict[str, str]:
    """[%cal] and [%csl] values for drawn arrows and circles."""
    cal = ",".join(SHAPE_LETTERS.get(c, "G") + square_name(a) + square_name(b) for (a, b), c in arrows.items())
    csl = ",".join(SHAPE_LETTERS.get(c, "G") + square_name(sq) for sq, c in circles.items())
    return {"cal": cal, "csl": csl}


def parse_shapes(cal: str, csl: str) -> tuple[dict[tuple[int, int], str], dict[int, str]]:
    """Arrows and circles from [%cal] and [%csl] values (bad items skipped)."""
    arrows: dict[tuple[int, int], str] = {}
    circles: dict[int, str] = {}
    for item in (cal + "," + csl).split(","):
        m = _SHAPE_RE.match(item.strip())
        if not m:
            continue
        color = _LETTER_SHAPES[m.group("color")]
        a = parse_square(m.group("a"))
        if m.group("b") and m.group("b") != m.group("a"):
            arrows[(a, parse_square(m.group("b")))] = color
        else:
            circles[a] = color
    return arrows, circles


class Node:
    """A position of the game tree: the start (the root, no move) or the
    position after `move`. The first child continues the line; the others
    are variations, alternatives to the first child."""

    def __init__(self, position: Position, move: Move | None = None, san: str = "", parent: Node | None = None):
        self.position = position
        self.move = move
        self.san = san
        self.parent = parent
        self.children: list[Node] = []
        self.ply = parent.ply + 1 if parent is not None else 0
        self.eval: MoveEval | None = None  # what the engine that played the move reported, and the time taken
        self.clock_ms: float | None = None  # the mover's time left after the move (clocked games)
        self.note = ""  # a match runner's notes ("tl=1.2s, nps=..."), kept on saving
        self.comment = ""  # the annotator's text after the move (the root's: before the first move)
        self.starting_comment = ""  # text before the move, at the start of a variation
        self.nags: list[int] = []
        self.arrows: dict[tuple[int, int], str] = {}  # drawn arrows: (from, to) -> color name
        self.circles: dict[int, str] = {}  # drawn circles: square -> color name
        self.review: PositionEval | None = None  # the game review's evaluation of this position

    def __repr__(self) -> str:
        return f"Node({self.ply}: {self.san or 'start'})"

    @property
    def is_root(self) -> bool:
        return self.parent is None

    def path(self) -> list[Node]:
        """The nodes from the root to this one."""
        out = []
        node: Node | None = self
        while node is not None:
            out.append(node)
            node = node.parent
        return out[::-1]

    def uci_moves(self) -> list[str]:
        """The moves from the start to this position, in UCI notation."""
        return [n.move.uci() for n in self.path()[1:]]

    def line(self) -> list[Node]:
        """The line through this node: the path to it, then its main
        continuation (first children) to the end."""
        out = self.path()
        node = self
        while node.children:
            node = node.children[0]
            out.append(node)
        return out

    def end(self) -> Node:
        """The last node of this node's line."""
        node = self
        while node.children:
            node = node.children[0]
        return node

    def child(self, move: Move) -> Node | None:
        for c in self.children:
            if c.move == move:
                return c
        return None

    def siblings(self) -> list[Node]:
        """The moves played from the same position (this one included)."""
        return self.parent.children if self.parent is not None else [self]

    def is_main(self) -> bool:
        """On the game's main line."""
        node = self
        while node.parent is not None:
            if node.parent.children[0] is not node:
                return False
            node = node.parent
        return True

    def starts_variation(self) -> bool:
        return self.parent is not None and self.parent.children[0] is not self

    def shapes(self) -> tuple[dict[tuple[int, int], str], dict[int, str]]:
        return dict(self.arrows), dict(self.circles)

    def set_shapes(self, arrows: dict[tuple[int, int], str], circles: dict[int, str]) -> None:
        self.arrows, self.circles = dict(arrows), dict(circles)


MoveRecord = Node  # the name main-line moves had before the tree


class GameRecord:
    """A game: the start position, a tree of moves (the main line and
    variations), the result and tags. `moves` (nodes) and `positions` are
    the main line, kept up to date by the methods that change the tree."""

    def __init__(self, start: Position | None = None, result: str = "*", reason: str = "",
                 tags: dict[str, str] | None = None):
        self.root = Node(start if start is not None else Position.start())
        self.result = result
        self.reason = reason
        self.tags: dict[str, str] = dict(tags or {})
        self.moves: list[Node] = []
        self.positions: list[Position] = [self.root.position]

    @property
    def start(self) -> Position:
        return self.root.position

    @property
    def position(self) -> Position:
        return self.positions[-1]

    @property
    def end(self) -> Node:
        """The main line's last node (the root if no moves)."""
        return self.moves[-1] if self.moves else self.root

    def _sync(self) -> None:
        """Re-reads the main line after the tree changed."""
        self.moves = self.root.line()[1:]
        self.positions = [self.root.position] + [n.position for n in self.moves]

    def contains(self, node: Node) -> bool:
        """Whether `node` is (still) in this tree."""
        while node.parent is not None:
            if not any(c is node for c in node.parent.children):
                return False
            node = node.parent
        return node is self.root

    # --- Changes ----------------------------------------------------------------------

    def push(self, move: Move, eval: MoveEval | None = None, clock_ms: float | None = None) -> Node:
        """Plays `move` at the end of the main line (a game's move)."""
        node = self._new_child(self.end, move)
        node.eval, node.clock_ms = eval, clock_ms
        self.moves.append(node)
        self.positions.append(node.position)
        return node

    def pop(self) -> Node:
        """Takes back the main line's last move (and any variations of it)."""
        node = self.moves[-1]
        self.truncate(len(self.moves) - 1)
        return node

    def truncate(self, ply: int) -> None:
        """Keeps the first `ply` moves of the main line; what follows them
        goes, variations included."""
        (self.moves[ply - 1] if ply > 0 else self.root).children.clear()
        self._sync()

    def add(self, parent: Node, move: Move) -> Node:
        """The node for `move` played after `parent`: the existing one, or a
        new variation (the continuation of the line, if `parent` has none).
        Raises IllegalMoveError for an illegal move."""
        node = parent.child(move)
        if node is None:
            node = self._new_child(parent, move)
            if node is parent.children[0]:
                self._sync()
        return node

    def add_line(self, parent: Node, moves: list[Move]) -> Node:
        """Plays `moves` one after the other from `parent` (see add); returns
        the last node."""
        node = parent
        for m in moves:
            node = self.add(node, m)
        return node

    def delete(self, node: Node) -> Node:
        """Removes `node` and what follows it; returns its parent. A
        variation's next one (if any) then continues the line."""
        if node.parent is None:
            raise ValueError("the start position can't be deleted")
        parent = node.parent
        parent.children = [c for c in parent.children if c is not node]
        self._sync()
        return parent

    def promote(self, node: Node) -> None:
        """Makes the variation `node` is in one step more important: it
        trades places with the line before it, where it branches off."""
        while node.parent is not None and node.parent.children[0] is node:
            node = node.parent  # up to where this line branches off
        if node.parent is None:
            return
        siblings = node.parent.children
        i = next(k for k, c in enumerate(siblings) if c is node)
        siblings[i - 1], siblings[i] = siblings[i], siblings[i - 1]
        self._sync()

    def make_main(self, node: Node) -> None:
        """Makes the line through `node` the main line."""
        while node.parent is not None:
            siblings = node.parent.children
            i = next(k for k, c in enumerate(siblings) if c is node)
            siblings.insert(0, siblings.pop(i))
            node = node.parent
        self._sync()

    def _new_child(self, parent: Node, move: Move) -> Node:
        san = parent.position.san(move)  # (checks legality)
        node = Node(parent.position.push(move), move, san, parent)
        parent.children.append(node)
        return node

    # --- Queries -----------------------------------------------------------------------

    def uci_moves(self, ply: int | None = None) -> list[str]:
        return [m.move.uci() for m in self.moves[:ply]]

    def repetitions(self, ply: int | None = None) -> int:
        """How often the position after `ply` main-line moves (default: now)
        occurred up to then, itself included."""
        ply = len(self.moves) if ply is None else ply
        return self.repetitions_at(self.moves[ply - 1] if ply > 0 else self.root)

    @staticmethod
    def repetitions_at(node: Node) -> int:
        """How often `node`'s position occurred on the way to it, itself included."""
        pos = node.position
        key = pos.key()
        count = 0
        n: Node | None = node
        for _ in range(pos.halfmove + 1):  # no position before the last irreversible move can repeat
            if n is None:
                break
            count += n.position.key() == key
            n = n.parent
        return count

    def ending(self) -> tuple[str, str] | None:
        """(result, reason) if the rules end the game now."""
        return self.ending_at(self.end)

    def ending_at(self, node: Node) -> tuple[str, str] | None:
        """(result, reason) if the rules end the game at `node`."""
        e = rules_ending(node.position, self.repetitions_at(node))
        return ending_result(node.position, e) if e else None

    def nodes(self) -> list[Node]:
        """Every node, depth first, main continuations before variations."""
        out, stack = [], [self.root]
        while stack:
            node = stack.pop()
            out.append(node)
            stack.extend(reversed(node.children))
        return out

    # --- PGN ---------------------------------------------------------------------------

    def _comment_of(self, node: Node) -> str:
        """The PGN comment for a node: the engine's evaluation (as match
        runners write it), notes, the annotator's text, the reason the game
        ended (after the main line's last move), then the commands."""
        parts = [node.eval.comment() if node.eval else "", node.note, node.comment]
        if self.reason and node is self.end and node is not self.root:
            parts.append(self.reason)
        commands = {}
        if node.clock_ms is not None:
            commands["clk"] = format_clock_command(node.clock_ms)
        commands.update(shapes_commands(node.arrows, node.circles))
        if node.review is not None:
            commands["eval"] = node.review.command()
        return join_commands(", ".join(p for p in parts if p), commands)

    def to_pgn(self) -> PgnGame:
        tags = dict(self.tags)
        tags["Result"] = self.result
        tags["PlyCount"] = str(len(self.moves))
        if self.reason:
            tags["Termination"] = termination_tag(self.reason)
        game = PgnGame(tags=tags, start=self.start, result=self.result)
        game.root.comment = self._comment_of(self.root)
        stack = [(self.root, game.root)]
        while stack:
            node, out = stack.pop()
            for child in node.children:
                c = out.add(child.move, self._comment_of(child))
                c.nags = list(child.nags)
                c.starting_comment = child.starting_comment
                stack.append((child, c))
        return game

    @staticmethod
    def from_pgn(game: PgnGame) -> GameRecord:
        """A record of a PGN game. A move's comment gives the engine's
        evaluation ("+0.35/12 1.2s", or the time alone), match runners' notes
        ("tl=1.2s"), the annotator's text and commands (clock, arrows and
        circles, review evaluations); after the main line's last move, text
        like "White mates" gives the reason the game ended (as fastchess and
        this GUI write it)."""
        record = GameRecord(game.start, result=game.result,
                            tags={k: v for k, v in game.tags.items() if k not in ("SetUp", "FEN")})
        mainline = game.mainline()
        last = mainline[-1] if mainline else None
        reason = record._read_comment(record.root, game.root.comment, False)
        stack = [(game.root, record.root)]
        while stack:
            src, node = stack.pop()
            for child in src.children:
                c = record._new_child(node, child.move)
                c.nags = list(child.nags)
                c.starting_comment = child.starting_comment
                found = record._read_comment(c, child.comment, child is last and game.result != "*")
                reason = found or reason
                stack.append((child, c))
        record._sync()
        termination = game.tags.get("Termination", "")
        record.reason = reason or (termination if termination not in ("", "normal") else "")
        return record

    @staticmethod
    def _read_comment(node: Node, comment: str, may_end: bool) -> str:
        """Fills `node` from its PGN comment; returns the reason the game
        ended if `may_end` and the comment gives one."""
        text, commands = split_commands(comment)
        parts = [p.strip() for p in text.split(",") if p.strip()]
        if parts and node.move is not None:
            node.eval = parse_eval_comment(parts[0])
            if node.eval is not None:
                parts = parts[1:]
        reason = ""
        if may_end:
            for i, p in enumerate(parts):
                if _REASON_RE.match(p):
                    reason = ", ".join(parts[i:])
                    parts = parts[:i]
                    break
        node.note = ", ".join(p for p in parts if _NOTE_RE.match(p))
        node.comment = ", ".join(p for p in parts if not _NOTE_RE.match(p))
        if "clk" in commands:
            node.clock_ms = parse_clock_command(commands["clk"])
        node.arrows, node.circles = parse_shapes(commands.get("cal", ""), commands.get("csl", ""))
        if "eval" in commands:
            node.review = parse_eval_command(commands["eval"])
        return reason


def termination_tag(reason: str) -> str:
    """The PGN Termination tag value for a reason text."""
    low = reason.lower()
    if "on time" in low:
        return "time forfeit"
    if "illegal" in low or "no move" in low:
        return "rules infraction"
    if "engine failed" in low or "aborted" in low:
        return "abandoned"
    return "normal"


class GameController(QObject):
    """Runs a game: whose turn it is, clocks, engine searches, the result."""

    changed = Signal()  # moves made or taken back, a new game, a game over
    game_over = Signal(str, str)  # result, reason
    engine_info = Signal(int, object)  # color whose engine reported, Info (current searches only)
    message = Signal(str)  # errors and notices
    status_changed = Signal(str)
    engine_log = Signal(str, str, str)  # engine name, direction (">", "<", "!"), line

    IDLE, PREPARING, RUNNING, OVER = "idle", "preparing", "running", "over"

    def __init__(self, parent: QObject | None = None, now=time.monotonic):
        super().__init__(parent)
        self._now = now
        self.record = GameRecord()
        self.profiles: list[EngineProfile | None] = [None, None]
        self.engines: list[UciEngine | None] = [None, None]
        # Engines kept between games (tournaments): profile key -> idle engines.
        # None: an engine no game needs any more is quit.
        self.engine_cache: dict[str, list[UciEngine]] | None = None
        # An opening book for the engines (anything with book_move(position, rng),
        # e.g. a database's position index), used while the game has fewer
        # than book_plies moves.
        self.book = None
        self.book_plies = 0
        self.rng = random.Random()
        self.tc = TimeControl()
        self.clock = GameClock(self.tc, now)
        self.status = self.IDLE
        self._search: int | None = None
        self._search_color: int | None = None
        self._last_info: Info | None = None
        self._flag = QTimer(self)
        self._flag.setSingleShot(True)
        self._flag.timeout.connect(self._check_flag)

    # --- Queries ------------------------------------------------------------------

    @property
    def human_to_move(self) -> bool:
        return self.status == self.RUNNING and self.profiles[self.record.position.turn] is None

    @property
    def thinking_color(self) -> int | None:
        return self._search_color if self._search is not None else None

    def player_name(self, color: int) -> str:
        """The profile's name, or the name the game record gives (loaded
        games), or "Human"."""
        p = self.profiles[color]
        if p is not None:
            return p.name
        return self.record.tags.get(COLOR_NAMES[color], "Human") or "Human"

    # --- Commands -----------------------------------------------------------------

    def new_game(self, white: EngineProfile | None, black: EngineProfile | None, tc: TimeControl,
                 start: Position | None = None, event: str = "ChessGPU GUI game") -> None:
        self._end_searches()
        self._flag.stop()
        self.tc = tc
        self.clock = GameClock(tc, self._now)
        start = start or Position.start()
        self.profiles = [white, black]
        self.record = GameRecord(start, tags={
            "Event": event, "Site": "ChessGPU GUI", "Date": date.today().strftime("%Y.%m.%d"), "Round": "-",
            "White": self.player_name(WHITE), "Black": self.player_name(BLACK), "TimeControl": tc.pgn_tag()})
        self._set_status(self.PREPARING)
        self._prepare_engines()
        self.changed.emit()
        self._maybe_begin()

    def load(self, record: GameRecord) -> None:
        """Shows a finished or loaded game (no players, nothing running)."""
        self._end_searches()
        self._flag.stop()
        self.clock = GameClock(TimeControl(), self._now)
        self.record = record
        self.profiles = [None, None]
        self._set_status(self.OVER if record.result != "*" else self.IDLE)
        self.changed.emit()

    def human_move(self, move: Move) -> bool:
        """Plays a move for the human to move; False if it isn't their turn
        or the move is illegal."""
        if not self.human_to_move or not self.record.position.is_legal(move):
            return False
        self._play(move, None)
        return True

    def explore(self, node: Node, move: Move) -> Node | None:
        """Outside a running game (an analysis board): plays `move` after
        `node`, as the move already there, a new variation, or the
        continuation of the line. Returns the move's node; None if a game
        runs, `node` isn't in the game or the move is illegal."""
        if self.status in (self.RUNNING, self.PREPARING) or not self.record.contains(node):
            return None
        if not node.position.is_legal(move):
            return None
        child = self.record.add(node, move)
        self.changed.emit()
        return child

    def edit(self, change, *args):
        """Changes the tree outside a running game (record.delete, promote,
        make_main, add_line...): returns what the change returns, None if a
        game runs."""
        if self.status in (self.RUNNING, self.PREPARING):
            return None
        result = change(*args)
        self.changed.emit()
        return result

    def resign(self, color: int | None = None) -> None:
        if self.status != self.RUNNING:
            return
        color = self.record.position.turn if color is None else color
        self._finish(win_for(other(color)), f"{COLOR_NAMES[color]} resigns")

    def abort(self) -> None:
        if self.status in (self.RUNNING, self.PREPARING):
            self._finish("*", "Game aborted")

    def move_now(self) -> None:
        """Makes the thinking engine play its best move so far."""
        color = self.thinking_color
        if color is not None and self.engines[color] is not None:
            self.engines[color].stop()

    def takeback(self) -> bool:
        """Takes back the last move, and the engine's reply before it, so a
        human is to move again; the clocks return to their times then. Only
        in games with a human player (also after the game ended)."""
        if self.status not in (self.RUNNING, self.OVER) or not self.record.moves:
            return False
        if self.profiles[WHITE] is not None and self.profiles[BLACK] is not None:
            return False
        self._end_searches()
        self._flag.stop()
        self.clock.stop()
        self.record.pop()
        while self.record.moves and self.profiles[self.record.position.turn] is not None:
            self.record.pop()
        for color in (WHITE, BLACK):
            left = float(self.tc.base_ms or 0)
            for m, pos in zip(self.record.moves, self.record.positions):
                if pos.turn == color and m.clock_ms is not None:
                    left = m.clock_ms
            self.clock.remaining[color] = left
        self.record.result, self.record.reason = "*", ""
        self._set_status(self.RUNNING)
        self.changed.emit()
        self._next_turn()
        return True

    def adjudicate(self, result: str, reason: str) -> bool:
        """Ends a running game with `result` (a tournament's adjudication)."""
        if self.status != self.RUNNING:
            return False
        self._finish(result, reason)
        return True

    def keep_engines(self, keep: bool) -> None:
        """Keep engines no game needs any more, for later games with their
        profile (no restart: a GPU engine's set-up takes a moment), or quit
        the kept ones."""
        if keep:
            if self.engine_cache is None:
                self.engine_cache = {}
            return
        for engines in (self.engine_cache or {}).values():
            for e in engines:
                e.quit()
        self.engine_cache = None

    def shutdown(self) -> None:
        """Ends any search and quits the engines (call before exiting)."""
        self._flag.stop()
        self._search = None
        for i, e in enumerate(self.engines):
            if e is not None:
                e.quit()
                self.engines[i] = None
        self.keep_engines(False)

    # --- Engines ------------------------------------------------------------------

    def _prepare_engines(self) -> None:
        """Engines for the new game's profiles: the last game's engine of the
        same profile (told "ucinewgame"), else a kept one, else a new one.
        Every engine is in place before any starts or is let go: their state
        changes call _maybe_begin, which must not see a half-built lineup."""
        old = [e for e in self.engines if e is not None]
        new: list[UciEngine | None] = [None, None]
        fresh = []  # (engine, profile) to start
        alive = (EngineState.STARTING, EngineState.IDLE, EngineState.SEARCHING)
        for color in (WHITE, BLACK):
            profile = self.profiles[color]
            if profile is None:
                continue
            key = profile.key()
            e = next((x for x in old if x.profile_key == key and x.state in alive and all(x is not y for y in new)),
                     None)
            if e is None and self.engine_cache is not None:
                kept = self.engine_cache.get(key, [])
                while kept and e is None:
                    candidate = kept.pop()
                    if candidate.state in alive:
                        e = candidate
                    else:
                        candidate.quit()
            if e is None:
                e = self._make_engine(profile, color)
                fresh.append((e, profile))
            e.name = f"{COLOR_NAMES[color]} engine ({profile.name})"
            new[color] = e
        self.engines = new
        for x in old:
            if all(x is not y for y in new):
                if self.engine_cache is not None and x.state in (EngineState.IDLE, EngineState.SEARCHING):
                    x.stop()
                    self.engine_cache.setdefault(x.profile_key, []).append(x)
                else:
                    x.quit()
        for e in new:
            if e is not None and all(e is not f for f, _ in fresh):
                e.new_game()
        for e, profile in fresh:
            e.start(profile.path, profile.args, profile.options)

    def _make_engine(self, profile: EngineProfile, color: int) -> UciEngine:
        """A new engine, not started yet; its signals reach the game for
        whichever color it plays at the time (engines move between colors
        and games)."""
        e = UciEngine(f"{COLOR_NAMES[color]} engine ({profile.name})", self)
        e.profile_key = profile.key()
        e.state_changed.connect(lambda _state: self._maybe_begin())
        e.info.connect(lambda sid, info, e=e: self._routed(e, self._on_info, sid, info))
        e.bestmove.connect(lambda sid, best, _ponder, e=e: self._routed(e, self._on_bestmove, sid, best))
        e.failed.connect(lambda msg, e=e: self._routed(e, self._on_engine_failed, msg))
        e.log.connect(lambda d, line, e=e: self.engine_log.emit(e.name, d, line))
        return e

    def _routed(self, engine: UciEngine, handler, *args) -> None:
        """Passes an engine's signal on with the color it plays (none: kept, ignored)."""
        for color in (WHITE, BLACK):
            if self.engines[color] is engine:
                handler(color, *args)
                return

    def _maybe_begin(self) -> None:
        if self.status != self.PREPARING:
            return
        if any(e is not None and (e.state != EngineState.IDLE or e.searching) for e in self.engines):
            return
        self._set_status(self.RUNNING)
        self._next_turn()

    def _end_searches(self) -> None:
        self._search = None
        self._search_color = None
        for e in self.engines:
            if e is not None and e.running:
                e.stop()

    def _on_info(self, color: int, sid: int, info: Info) -> None:
        if sid != self._search or color != self._search_color:
            return
        if info.score is not None and info.depth is not None:
            self._last_info = info
        self.engine_info.emit(color, info)

    def _on_bestmove(self, color: int, sid: int, best: str | None) -> None:
        if sid != self._search or color != self._search_color or self.status != self.RUNNING:
            return
        self._search = None
        if best is None:
            self._finish(win_for(other(color)), f"{COLOR_NAMES[color]} returned no move")
            return
        try:
            move = self.record.position.parse_uci(best)
        except IllegalMoveError:
            self._finish(win_for(other(color)), f"{COLOR_NAMES[color]} makes an illegal move: {best}")
            return
        info = self._last_info
        self._play(move, MoveEval(info.score, info.depth) if info else None)

    def _on_engine_failed(self, color: int, message: str) -> None:
        self.message.emit(message)
        if self.status == self.RUNNING:
            self._finish(win_for(other(color)), f"{COLOR_NAMES[color]}'s engine failed ({message})")
        elif self.status == self.PREPARING:
            self._finish("*", f"Game not started: {message}")

    # --- Turns --------------------------------------------------------------------

    def _set_status(self, status: str) -> None:
        if status != self.status:
            self.status = status
            self.status_changed.emit(status)

    def _next_turn(self) -> None:
        if self.status != self.RUNNING:
            return
        ending = self.record.ending()
        if ending:
            self._finish(*ending)
            return
        color = self.record.position.turn
        self.clock.start(color)
        if self.tc.clocked:
            self._arm_flag()
        engine = self.engines[color]
        book = self._book_move() if engine is not None else None
        if book is not None:
            self._search, self._search_color = None, None
            n = len(self.record.moves)
            QTimer.singleShot(BOOK_DELAY_MS, lambda m=book, c=color, n=n: self._play_book(m, c, n))
        elif engine is not None:
            self._last_info = None
            self._search_color = color
            limits = self.tc.engine_limits(self.clock.left(WHITE), self.clock.left(BLACK))
            self._search = -1  # no real id matches while go() runs
            self._search = engine.go(self.record.start.fen(), self.record.uci_moves(), **limits)
        self.changed.emit()

    def _book_move(self) -> Move | None:
        if self.book is None or len(self.record.moves) >= self.book_plies:
            return None
        return self.book.book_move(self.record.position, self.rng)

    def _play_book(self, move: Move, color: int, moves: int) -> None:
        """An engine's move from the book (if the game is still where it was)."""
        if self.status != self.RUNNING or len(self.record.moves) != moves or self.record.position.turn != color:
            return
        self._play(move, None, comment="book")  # (as cutechess writes book moves)

    def _play(self, move: Move, eval: MoveEval | None, comment: str = "") -> None:
        color = self.record.position.turn
        self._flag.stop()
        elapsed = self.clock.stop()
        if self.tc.clocked and self.clock.remaining[color] < 0:
            self._time_forfeit(color)
            return
        if self.tc.clocked:
            self.clock.add_increment(color)
        eval = MoveEval(eval.score if eval else None, eval.depth if eval else None, elapsed / 1000)
        node = self.record.push(move, eval, self.clock.remaining[color] if self.tc.clocked else None)
        node.comment = comment
        self.changed.emit()
        self._next_turn()

    def _arm_flag(self) -> None:
        if self.clock.running is not None:
            self._flag.start(max(0, int(self.clock.left(self.clock.running)) + 1))

    def _check_flag(self) -> None:
        if self.status != self.RUNNING or self.clock.running is None:
            return
        color = self.clock.running
        if self.clock.left(color) > 0:
            self._arm_flag()  # woke up early
            return
        self.clock.stop()
        self._time_forfeit(color)

    def _time_forfeit(self, color: int) -> None:
        winner = other(color)
        pieces = self.record.position.board
        lone_king = sum(1 for p in pieces if p != "." and (p.isupper() == (winner == WHITE))) == 1
        if lone_king:
            self._finish("1/2-1/2", f"{COLOR_NAMES[color]} loses on time, but {COLOR_NAMES[winner]} cannot mate: draw")
        else:
            self._finish(win_for(winner), f"{COLOR_NAMES[color]} loses on time")

    def _finish(self, result: str, reason: str) -> None:
        self._flag.stop()
        self.clock.stop()
        self._end_searches()
        self.record.result = result
        self.record.reason = reason
        self._set_status(self.OVER)
        self.changed.emit()
        self.game_over.emit(result, reason)


class AnalysisController(QObject):
    """Infinite analysis of one position at a time by one engine, with
    `lines` principal variations (the engine's MultiPV option, if it has
    one); also short searches for hints (when not analysing)."""

    info = Signal(object)  # Info about the analysed position
    message = Signal(str)
    running_changed = Signal(bool)
    engine_log = Signal(str, str, str)  # engine name, direction, line
    hint_ready = Signal(object, object)  # best move (UCI, or None), the last scored Info (or None)

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self.engine: UciEngine | None = None
        self.profile: EngineProfile | None = None
        self.lines = 1
        self._lines_pending = False  # set_lines while the engine started: applied once it's ready
        self._search: int | None = None
        self._target: tuple[str, list[str]] | None = None  # (start FEN, moves) to analyse
        self._hint_search: int | None = None
        self._hint_info: Info | None = None

    @property
    def running(self) -> bool:
        return self._target is not None

    def set_profile(self, profile: EngineProfile) -> None:
        if self.engine is not None and self.profile is not None and self.profile.key() == profile.key() \
                and self.engine.state in (EngineState.STARTING, EngineState.IDLE, EngineState.SEARCHING):
            return
        if self.engine is not None:
            self.engine.quit()
        self.profile = profile
        self._hint_search = None
        self._lines_pending = False
        e = UciEngine(f"analysis engine ({profile.name})", self)
        e.state_changed.connect(lambda _s: self._resume())
        e.info.connect(self._on_info)
        e.bestmove.connect(self._on_bestmove)
        e.failed.connect(self._on_failed)
        e.ready.connect(self._on_ready)
        e.log.connect(lambda d, line, n=e.name: self.engine_log.emit(n, d, line))
        self.engine = e
        options = dict(profile.options)
        if self.lines > 1:
            options["MultiPV"] = str(self.lines)  # (ignored by engines without it)
        e.start(profile.path, profile.args, options)

    def set_lines(self, n: int) -> None:
        """How many lines to analyse (engines with a MultiPV option)."""
        n = max(1, n)
        if n == self.lines:
            return
        self.lines = n
        if self.engine is None:
            return
        if self.engine.state == EngineState.STARTING:
            self._lines_pending = True
        else:
            self._apply_lines()

    def has_multipv(self) -> bool:
        return self.engine is not None and self.engine.option("MultiPV") is not None

    def _apply_lines(self) -> None:
        if self.engine is None or not self.engine.set_option("MultiPV", self.lines):
            return
        if self.running:  # analyse again with the new count (the option goes out first)
            self._search = None
            self._resume()

    def _on_ready(self) -> None:
        if self._lines_pending:
            self._lines_pending = False
            self._apply_lines()

    def hint(self, start_fen: str, moves: list[str], movetime_ms: int) -> bool:
        """A short search for a hint, when not analysing; hint_ready follows."""
        if self.running or self.engine is None or self.engine.state in (EngineState.FAILED, EngineState.STOPPED):
            return False
        self._hint_info = None
        self._hint_search = -1  # (no real id matches while go() runs)
        self._hint_search = self.engine.go(start_fen, moves, movetime=movetime_ms)
        return True

    def analyze(self, start_fen: str, moves: list[str]) -> None:
        """Analyses the position after `moves` from `start_fen` (replacing
        any previous analysis); nothing for a position without moves."""
        pos = Position.from_fen(start_fen)
        for m in moves:
            pos = pos.push(Move.from_uci(m))
        if not pos.legal_moves():
            self.stop()
            return
        was = self.running
        self._target = (start_fen, list(moves))
        self._search = None
        if not was:
            self.running_changed.emit(True)
        self._resume()

    def stop(self) -> None:
        was = self.running
        self._target = None
        self._search = None
        if self.engine is not None and self.engine.running:
            self.engine.stop()
        if was:
            self.running_changed.emit(False)

    def shutdown(self) -> None:
        self.stop()
        if self.engine is not None:
            self.engine.quit()
            self.engine = None

    def _resume(self) -> None:
        if self._target is None or self._search is not None or self.engine is None:
            return
        if self.engine.state not in (EngineState.IDLE, EngineState.SEARCHING):
            return
        fen, moves = self._target
        self._search = -1  # starting: a re-entrant call (via state_changed) must not start another
        self._search = self.engine.go(fen, moves, infinite=True)

    def _on_info(self, sid: int, info: Info) -> None:
        if sid == self._search:
            self.info.emit(info)
        elif sid == self._hint_search and info.score is not None and info.bound is None and (info.multipv or 1) == 1:
            self._hint_info = info

    def _on_bestmove(self, sid: int, best: str | None, _ponder) -> None:
        if sid == self._hint_search:
            self._hint_search = None
            if not self.running:  # (an analysis started meanwhile ended the hint early)
                self.hint_ready.emit(best, self._hint_info)

    def _on_failed(self, message: str) -> None:
        self.message.emit(message)
        self.stop()
