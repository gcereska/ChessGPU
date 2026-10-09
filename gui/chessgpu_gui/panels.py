"""Side panels: player cards (name, clock, material), move list, navigation
buttons, engine output, eval bar and eval graph. Scores shown here are from
White's point of view."""

from __future__ import annotations

import html

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (QAbstractItemView, QFrame, QGridLayout, QHBoxLayout, QHeaderView, QLabel,
                               QSizePolicy, QTableWidget, QTableWidgetItem, QToolButton, QVBoxLayout, QWidget)

from .game import GameController, GameRecord, Node
from .icons import icon
from .pieces import PieceSet
from .rules import BLACK, WHITE, IllegalMoveError, Position
from .score import Score
from .theme import current_theme
from .uci import Info

PIECE_VALUES = {"Q": 9, "R": 5, "B": 3, "N": 3, "P": 1}
FIGURINES = {"K": "♔", "Q": "♕", "R": "♖", "B": "♗", "N": "♘"}
LOW_TIME_MS = 20_000  # a running clock below this turns red


def white_score(score: Score | None, side_to_move: int) -> Score | None:
    """A score from the side to move's point of view, seen from White's."""
    if score is None:
        return None
    return score if side_to_move == WHITE else score.negated()


def node_score(node: Node) -> Score | None:
    """White's-view score of a position in the tree: the game review's, or
    else what the engine that played the move into it reported."""
    if node.review is not None:
        return node.review.score
    if node.parent is not None and node.eval is not None and node.eval.score is not None:
        return white_score(node.eval.score, node.parent.position.turn)
    return None


def win_probability(score: Score | None) -> float:
    """0..1 for White, from a White's-view score (a logistic of centipawns)."""
    if score is None:
        return 0.5
    if score.mate is not None:
        return 1.0 if score.winning() else 0.0
    return 1.0 / (1.0 + 10 ** (-score.cp / 400))


def format_clock(ms: float) -> str:
    """h:mm:ss from an hour, m:ss from 10 s, s.t below (0.0 when out of time)."""
    ms = max(0.0, ms)
    if ms >= 3_600_000:
        s = int(ms // 1000)
        return f"{s // 3600}:{s // 60 % 60:02d}:{s % 60:02d}"
    if ms >= 10_000:
        s = int(ms // 1000)
        return f"{s // 60}:{s % 60:02d}"
    return f"{int(ms // 1000)}.{int(ms // 100) % 10}"


def figurine_san(san: str) -> str:
    """SAN with figurines for the pieces (Nf3 -> ♘f3, e8=Q -> e8=♕)."""
    if san and san[0] in FIGURINES:
        san = FIGURINES[san[0]] + san[1:]
    if "=" in san:
        i = san.index("=")
        if i + 1 < len(san) and san[i + 1] in FIGURINES:
            san = san[:i + 1] + FIGURINES[san[i + 1]] + san[i + 2:]
    return san


def pv_san(root: Position, pv: list[str], limit: int = 24, figurines: bool = False) -> str:
    """A UCI principal variation as numbered SAN; stops at an illegal move."""
    parts = []
    pos = root
    for i, text in enumerate(pv[:limit]):
        try:
            move = pos.parse_uci(text)
        except IllegalMoveError:
            break
        if pos.turn == WHITE:
            parts.append(f"{pos.fullmove}.")
        elif i == 0:
            parts.append(f"{pos.fullmove}...")
        san = pos.san(move)
        parts.append(figurine_san(san) if figurines else san)
        pos = pos.push(move)
    if len(pv) > limit and len(parts) >= limit:
        parts.append("…")
    return " ".join(parts)


def material(position: Position) -> tuple[dict[int, list[str]], int]:
    """What each side is up, as on lichess: per color, the piece kinds it has
    more of than the other side (most valuable first), and White's material
    lead in pawns (negative: Black leads)."""
    counts = {WHITE: dict.fromkeys(PIECE_VALUES, 0), BLACK: dict.fromkeys(PIECE_VALUES, 0)}
    for piece in position.board:
        kind = piece.upper()
        if kind in PIECE_VALUES:
            counts[WHITE if piece.isupper() else BLACK][kind] += 1
    ahead = {WHITE: [], BLACK: []}
    for kind in PIECE_VALUES:
        diff = counts[WHITE][kind] - counts[BLACK][kind]
        side = WHITE if diff > 0 else BLACK
        ahead[side] += [kind] * abs(diff)
    lead = sum(PIECE_VALUES[k] * (counts[WHITE][k] - counts[BLACK][k]) for k in PIECE_VALUES)
    return ahead, lead


class MaterialStrip(QWidget):
    """Small pieces of the kinds one side is up, and its lead (+N)."""

    def __init__(self, piece_set: PieceSet, parent: QWidget | None = None):
        super().__init__(parent)
        self.piece_set = piece_set
        self.kinds: list[str] = []
        self.color = WHITE  # whose lead this shows: the pieces drawn are the opponent's
        self.lead = 0
        self.setMinimumHeight(18)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_material(self, color: int, kinds: list[str], lead: int) -> None:
        self.color, self.kinds, self.lead = color, kinds, lead
        self.update()

    def text(self) -> str:
        return f"+{self.lead}" if self.lead > 0 else ""

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        size = 18
        x = 0.0
        previous = None
        for kind in self.kinds:
            x += 4 if previous is not None and previous != kind else 0
            piece = kind.lower() if self.color == WHITE else kind  # the opponent's pieces
            p.drawPixmap(QPointF(x, 0), self.piece_set.pixmap(piece, size, self.devicePixelRatioF()))
            x += size * 0.55
            previous = kind
        if self.text():
            p.setPen(QColor(current_theme().text_dim))
            font = QFont(self.font())
            font.setPixelSize(12)
            p.setFont(font)
            p.drawText(QRectF(x + (8 if self.kinds else 0), 0, 60, size), Qt.AlignmentFlag.AlignVCenter, self.text())
        p.end()


class PlayerCard(QFrame):
    """One player: kind (engine or human), name, what they're doing, their
    material lead and their clock. Shows `color`'s player."""

    def __init__(self, controller: GameController, piece_set: PieceSet, color: int, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("playerCard")
        self.controller = controller
        self.color = color
        self.icon = QLabel()
        self.icon.setFixedSize(28, 28)
        self.name = QLabel()
        self.name.setObjectName("playerName")
        self.sub = QLabel()
        self.sub.setObjectName("playerSub")
        self.material = MaterialStrip(piece_set)
        self.clock = QLabel("–")
        self.clock.setObjectName("clock")
        self.clock.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.clock.setMinimumWidth(110)
        text = QVBoxLayout()
        text.setSpacing(1)
        title = QHBoxLayout()
        title.setSpacing(8)
        title.addWidget(self.name)
        title.addWidget(self.sub)
        title.addStretch()
        text.addLayout(title)
        text.addWidget(self.material)
        row = QHBoxLayout(self)
        row.setContentsMargins(10, 6, 8, 6)
        row.addWidget(self.icon)
        row.addLayout(text, 1)
        row.addWidget(self.clock)
        self._position: Position | None = None
        self._kind = ""
        self.refresh()

    def set_color(self, color: int) -> None:
        self.color = color
        self.refresh()

    def set_piece_set(self, piece_set: PieceSet) -> None:
        self.material.piece_set = piece_set
        self.material.update()

    def set_position(self, position: Position) -> None:
        """The position whose material to show (the one on the board)."""
        self._position = position
        ahead, lead = material(position)
        mine = lead if self.color == WHITE else -lead
        self.material.set_material(self.color, ahead[self.color], max(0, mine))

    def clock_state(self) -> str:
        g = self.controller
        if g.status != g.RUNNING or g.clock.running != self.color:
            return ""
        return "low" if g.tc.clocked and g.clock.left(self.color) < LOW_TIME_MS else "active"

    def refresh(self) -> None:
        g = self.controller
        color = self.color
        engine = g.profiles[color] is not None
        kind = "engine" if engine else "human"
        if kind != self._kind:
            self._kind = kind
            self.icon.setPixmap(icon(kind, current_theme().text_dim).pixmap(QSize(24, 24)))
        self.name.setText(g.player_name(color))
        if g.status == g.RUNNING and g.thinking_color == color:
            sub = "thinking…"
        elif g.status == g.RUNNING and g.record.position.turn == color:
            sub = "to move"
        else:
            sub = "White" if color == WHITE else "Black"
        self.sub.setText(sub)
        clocked = g.tc.clocked
        self.clock.setVisible(clocked)
        if clocked:
            self.clock.setText(format_clock(g.clock.left(color)))
        state = self.clock_state()
        if self.clock.property("state") != state:  # repolishing every tick is costly
            self.clock.setProperty("state", state)
            self.clock.style().unpolish(self.clock)
            self.clock.style().polish(self.clock)
        active = "true" if state else "false"
        if self.property("active") != active:
            self.setProperty("active", active)
            self.style().unpolish(self)
            self.style().polish(self)

    def retheme(self) -> None:
        self._kind = ""
        self.refresh()


class BoardArea(QWidget):
    """The eval bar and board, square, with a player card above and below
    exactly as wide as the bar and board together; centered in the space."""

    GAP = 6
    MARGIN = 8

    def __init__(self, board: QWidget, eval_bar: QWidget, top: QWidget, bottom: QWidget,
                 parent: QWidget | None = None):
        super().__init__(parent)
        self.board, self.eval_bar = board, eval_bar
        self.top, self.bottom = top, bottom
        for w in (board, eval_bar, top, bottom):
            w.setParent(self)

    def set_cards(self, top: QWidget, bottom: QWidget) -> None:
        self.top, self.bottom = top, bottom
        self._arrange()

    def sizeHint(self) -> QSize:
        return QSize(720, 820)

    def minimumSizeHint(self) -> QSize:
        return QSize(320, 420)

    def card_height(self) -> int:
        return max(self.top.sizeHint().height(), self.bottom.sizeHint().height(), 48)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        # As wide as the board this height allows, no wider: spare width
        # goes to the panels beside it.
        gap, margin = self.GAP, self.MARGIN
        side = max(64, (self.height() - 2 * margin - 2 * (self.card_height() + gap)) // 8 * 8)
        ideal = 2 * margin + self.eval_bar.width() + gap + side
        if self.maximumWidth() != ideal:
            self.setMaximumWidth(ideal)
        self._arrange()

    def _arrange(self) -> None:
        gap, margin = self.GAP, self.MARGIN
        bar = self.eval_bar.width()
        card = self.card_height()
        avail_w = self.width() - 2 * margin - bar - gap
        avail_h = self.height() - 2 * margin - 2 * (card + gap)
        side = max(64, int(min(avail_w, avail_h) // 8 * 8))
        total_w = bar + gap + side
        x = (self.width() - total_w) // 2
        y = (self.height() - (side + 2 * (card + gap))) // 2
        self.top.setGeometry(x, y, total_w, card)
        self.eval_bar.setGeometry(x, y + card + gap, bar, side)
        self.board.setGeometry(x + bar + gap, y + card + gap, side, side)
        self.bottom.setGeometry(x, y + card + gap + side + gap, total_w, card)


class NavigationBar(QWidget):
    """First, previous, next and last move buttons."""

    first = Signal()
    previous = Signal()
    next = Signal()
    last = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        self.buttons: dict[str, QToolButton] = {}
        for name, tip in (("first", "Start (Home)"), ("previous", "Back (Left)"), ("next", "Forward (Right)"),
                          ("last", "End (End)")):
            b = QToolButton()
            b.setToolTip(tip)
            b.setIconSize(QSize(20, 20))
            b.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            b.clicked.connect(getattr(self, name).emit)
            self.buttons[name] = b
            row.addWidget(b)
        self.retheme()

    def retheme(self) -> None:
        for name, b in self.buttons.items():
            b.setIcon(icon(name, current_theme().text))


class MoveListView(QTableWidget):
    """The moves in rows of two (White, Black); clicking a move selects the
    position after it."""

    ply_selected = Signal(int)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(0, 2, parent)
        self.setHorizontalHeaderLabels(["White", "Black"])
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.verticalHeader().setDefaultSectionSize(24)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setAlternatingRowColors(True)
        self.setShowGrid(False)
        self.figurines = False
        self._record: GameRecord | None = None
        self._offset = 0  # 1 when the game starts with a black move
        self.cellClicked.connect(self._clicked)

    def set_record(self, record: GameRecord, current_ply: int) -> None:
        self._record = record
        self._offset = 1 if record.start.turn == BLACK else 0
        slots = len(record.moves) + self._offset
        rows = (slots + 1) // 2
        self.setRowCount(rows)
        self.setVerticalHeaderLabels([f"{record.start.fullmove + r}." for r in range(rows)])
        for r in range(rows):
            for c in range(2):
                i = 2 * r + c - self._offset
                text, tip = "", ""
                if 0 <= i < len(record.moves):
                    m = record.moves[i]
                    text = figurine_san(m.san) if self.figurines else m.san
                    if m.eval and m.eval.score is not None:
                        tip = m.eval.comment()
                    elif m.eval and m.eval.seconds is not None:
                        tip = f"{m.eval.seconds:.1f} s"
                elif i < 0:
                    text = "…"
                item = QTableWidgetItem(text)
                item.setToolTip(tip)
                self.setItem(r, c, item)
        self.set_current(current_ply)

    def cell_of(self, ply: int) -> tuple[int, int] | None:
        """The cell of the move that leads to `ply` (None for ply 0)."""
        if ply <= 0:
            return None
        slot = ply - 1 + self._offset
        return slot // 2, slot % 2

    def set_current(self, ply: int) -> None:
        self.clearSelection()
        cell = self.cell_of(ply)
        if cell is not None and cell[0] < self.rowCount():
            self.setCurrentCell(*cell)
            self.scrollToItem(self.item(*cell))
        else:
            self.setCurrentCell(-1, -1)

    def _clicked(self, row: int, col: int) -> None:
        i = 2 * row + col - self._offset
        if self._record is not None and 0 <= i < len(self._record.moves):
            self.ply_selected.emit(i + 1)


class EngineOutput(QFrame):
    """One engine's latest search report: score, depth, nodes, speed and
    time, and its principal variations (several with MultiPV, best first).
    Clicking a move of a line asks to play the line up to that move."""

    pv_clicked = Signal(int, int)  # line (1: the best), how many of its moves
    PV_PLIES = 16  # moves shown per line

    def __init__(self, title: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("engineOutput")
        self.title = QLabel(title)
        self.title.setObjectName("outputTitle")
        self.header = QHBoxLayout()  # the title, and room for controls (the analysis's line count)
        self.header.setContentsMargins(0, 0, 0, 0)
        self.header.addWidget(self.title, 1)
        self.fields = {name: QLabel("–") for name in ("Score", "Depth", "Nodes", "NPS", "Time")}
        grid = QGridLayout()
        grid.setVerticalSpacing(0)
        for i, (name, label) in enumerate(self.fields.items()):
            caption = QLabel(name)
            caption.setObjectName("fieldName")
            grid.addWidget(caption, 0, i)
            label.setObjectName("fieldValue")
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            grid.addWidget(label, 1, i)
        self.lines_box = QVBoxLayout()
        self.lines_box.setSpacing(2)
        self.line_labels: list[QLabel] = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.addLayout(self.header)
        layout.addLayout(grid)
        layout.addLayout(self.lines_box)
        self.figurines = False
        self.max_lines = 1
        self.score: Score | None = None  # the best line's, White's view
        self.best: str | None = None  # the best line's first move (UCI)
        self.lines: dict[int, tuple[Score | None, list[str]]] = {}  # rank -> (White's-view score, UCI moves)
        self.root: Position | None = None  # where the lines start

    def clear(self, title: str | None = None) -> None:
        if title is not None:
            self.title.setText(title)
        for label in self.fields.values():
            label.setText("–")
        self.score, self.best = None, None
        self.lines = {}
        self._show_lines()

    def set_max_lines(self, n: int) -> None:
        self.max_lines = max(1, n)
        self.lines = {k: v for k, v in self.lines.items() if k <= self.max_lines}
        self._show_lines()

    def show_info(self, info: Info, root: Position) -> None:
        """Updates the fields this info line carries; the rest keep their value."""
        rank = info.multipv or 1
        if root != self.root:
            self.root, self.lines = root, {}
        exact = info.score is not None and info.bound is None
        if exact and rank == 1:
            self.score = white_score(info.score, root.turn)
            self.fields["Score"].setText(self.score.text())
        if info.depth is not None:
            self.fields["Depth"].setText(str(info.depth) + (f"/{info.seldepth}" if info.seldepth else ""))
        if info.nodes is not None:
            self.fields["Nodes"].setText(_count(info.nodes))
        if info.nps is not None:
            self.fields["NPS"].setText(_count(info.nps))
        if info.time_ms is not None:
            self.fields["Time"].setText(f"{info.time_ms / 1000:.1f} s")
        if info.pv and rank <= self.max_lines:
            score = white_score(info.score, root.turn) if exact else self.lines.get(rank, (None, []))[0]
            self.lines[rank] = (score, list(info.pv))
            if rank == 1:
                self.best = info.pv[0]
            self._show_lines()

    def line_text(self, rank: int) -> str:
        """A line's moves as numbered SAN (plain text)."""
        if rank not in self.lines or self.root is None:
            return ""
        return pv_san(self.root, self.lines[rank][1], self.PV_PLIES, self.figurines)

    def _show_lines(self) -> None:
        ranks = sorted(k for k in self.lines if k <= self.max_lines)
        while len(self.line_labels) < len(ranks):
            label = QLabel()
            label.setTextFormat(Qt.TextFormat.RichText)
            label.setTextInteractionFlags(Qt.TextInteractionFlag.LinksAccessibleByMouse)
            label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            label.linkActivated.connect(self._link)
            self.lines_box.addWidget(label)
            self.line_labels.append(label)
        for i, label in enumerate(self.line_labels):
            label.setVisible(i < len(ranks))
            if i < len(ranks):
                label.setText(self._line_html(ranks[i]))

    def _line_html(self, rank: int) -> str:
        t = current_theme()
        score, moves = self.lines[rank]
        parts = []
        pos = self.root
        for j, text in enumerate(moves[:self.PV_PLIES]):
            try:
                move = pos.parse_uci(text)
            except IllegalMoveError:
                break
            number = f"{pos.fullmove}." if pos.turn == WHITE else (f"{pos.fullmove}..." if j == 0 else "")
            san = pos.san(move)
            san = html.escape(figurine_san(san) if self.figurines else san)
            link = f'<a href="{rank}:{j + 1}" style="color:{t.text}; text-decoration:none">{san}</a>'
            parts.append(f'<span style="color:{t.text_dim}">{number}</span> {link}' if number else link)
            pos = pos.push(move)
        if len(moves) > self.PV_PLIES and len(parts) == self.PV_PLIES:
            parts.append("…")
        head = f"<b>{html.escape(score.text())}</b>&nbsp;&nbsp;" if score is not None else ""
        return head + " ".join(parts)

    def _link(self, href: str) -> None:
        rank, _, count = href.partition(":")
        if rank.isdigit() and count.isdigit():
            self.pv_clicked.emit(int(rank), int(count))


def _count(n: int) -> str:
    for unit, size in (("G", 1e9), ("M", 1e6), ("k", 1e3)):
        if n >= size:
            return f"{n / size:.1f}{unit}"
    return str(n)


class EvalBar(QWidget):
    """White's winning chances as a vertical bar (White at the bottom unless flipped)."""

    def __init__(self, board: QWidget | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        self.setFixedWidth(26)
        self.board = board  # a BoardView beside the bar: the bar spans its squares
        self.score: Score | None = None
        self.flipped = False

    def set_score(self, score: Score | None) -> None:
        self.score = score
        self.setToolTip(score.text() if score else "")
        self.update()

    def set_flipped(self, flipped: bool) -> None:
        self.flipped = flipped
        self.update()

    def white_fraction(self) -> float:
        return win_probability(self.score)

    def span(self) -> tuple[float, float]:
        """(top, height) of the bar: the board's squares, or the whole widget."""
        if self.board is not None:
            r = self.board.board_rect()
            top = self.board.mapTo(self.window(), r.topLeft().toPoint()).y() - \
                self.mapTo(self.window(), self.rect().topLeft()).y()
            return max(0.0, float(top)), min(float(self.height()), r.height())
        return 0.0, float(self.height())

    def paintEvent(self, event) -> None:
        t = current_theme()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        top, h = self.span()
        w = self.width()
        white_h = h * self.white_fraction()
        clip = QPainterPath()
        clip.addRoundedRect(QRectF(0, top, w, h), 4, 4)
        p.setClipPath(clip)
        p.fillRect(QRectF(0, top, w, h), QColor(t.eval_black))
        y = top if self.flipped else top + h - white_h
        p.fillRect(QRectF(0, y, w, white_h), QColor(t.eval_white))
        p.setPen(QPen(QColor(t.text_dim)))
        p.drawLine(QPointF(0, top + h / 2), QPointF(w, top + h / 2))
        if self.score is not None:
            text = self.score.text().lstrip("+")
            font = QFont(self.font())
            font.setPixelSize(10)
            font.setBold(True)
            p.setFont(font)
            winning_white = self.white_fraction() >= 0.5
            at_bottom = winning_white != self.flipped
            p.setPen(QColor("#202020") if winning_white else QColor("#f0f0f0"))
            rect = QRectF(0, top + h - 18, w, 16) if at_bottom else QRectF(0, top + 2, w, 16)
            p.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)
        p.end()


class EvalGraph(QWidget):
    """The evaluation after every move of a line (White's winning chances);
    clicking selects that position."""

    ply_selected = Signal(int)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setMinimumHeight(70)
        self.points: list[float | None] = []  # per ply 1..n: White's chances, None if unknown
        self.current = 0
        self.marks: dict[int, str] = {}  # ply -> color of a dot (the review's mistakes)

    def set_marks(self, marks: dict[int, str]) -> None:
        self.marks = dict(marks)
        self.update()

    def set_record(self, record: GameRecord, current_ply: int) -> None:
        """The game's main line."""
        self.set_line(record.root.line(), current_ply)

    def set_line(self, nodes: list[Node], current_ply: int) -> None:
        """A line from the start (nodes[0], the root) to its end."""
        self.points = []
        for node in nodes[1:]:
            s = node_score(node)
            self.points.append(win_probability(s) if s is not None else None)
        self.current = current_ply
        self.update()

    def x_of(self, ply: int) -> float:
        n = max(1, len(self.points))
        return 4 + (self.width() - 8) * ply / n

    def ply_at(self, x: float) -> int:
        n = max(1, len(self.points))
        return max(0, min(len(self.points), round((x - 4) / max(1, self.width() - 8) * n)))

    def paintEvent(self, event) -> None:
        t = current_theme()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor(t.graph))
        p.setPen(QPen(QColor(t.border), 1, Qt.PenStyle.DashLine))
        p.drawLine(0, h // 2, w, h // 2)
        known = [(i + 1, v) for i, v in enumerate(self.points) if v is not None]
        if known:
            known.insert(0, (0, 0.5))  # the game starts even
            area = QPainterPath(QPointF(self.x_of(known[0][0]), h))
            for ply, v in known:
                area.lineTo(self.x_of(ply), h * (1 - v))
            area.lineTo(self.x_of(known[-1][0]), h)
            area.closeSubpath()
            fill = QColor(t.graph_fill)
            fill.setAlpha(170)
            p.fillPath(area, fill)
            p.setPen(QPen(QColor(t.graph_fill), 1.5))
            for (a, va), (b, vb) in zip(known, known[1:]):
                p.drawLine(QPointF(self.x_of(a), h * (1 - va)), QPointF(self.x_of(b), h * (1 - vb)))
        p.setPen(QPen(QColor(t.accent), 2))
        x = self.x_of(self.current)
        p.drawLine(QPointF(x, 0), QPointF(x, h))
        p.setPen(QPen(QColor(t.graph), 1.5))
        for ply, color in self.marks.items():
            if 1 <= ply <= len(self.points) and self.points[ply - 1] is not None:
                p.setBrush(QColor(color))
                p.drawEllipse(QPointF(self.x_of(ply), h * (1 - self.points[ply - 1])), 4.5, 4.5)
        p.end()

    def mousePressEvent(self, event) -> None:
        if self.points:
            self.ply_selected.emit(self.ply_at(event.position().x()))
