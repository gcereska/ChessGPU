"""The board widget: draws a position with a piece set and a board theme,
and turns clicks and drags into legal moves.

- Moves slide into place (castling moves the rook too, captures fade, a
  promoting pawn turns into its piece at the end): plan_animation() works
  out the motion from the positions before and after, forward or back.
- The right mouse button draws shapes, as on lichess: a click toggles a
  circle on a square, a drag toggles an arrow; Shift, Ctrl and Alt pick red,
  blue and yellow instead of green. A left click clears them.
- Premoves: while an engine thinks, the human's side (premove_color) can
  queue one move, shown in blue; whoever owns the board plays it when the
  turn comes, if it's legal then. A right click cancels it.
- Promotions ask with an overlay on the board (or queen automatically).
In edit mode the board only reports clicked squares (the board editor).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from PySide6.QtCore import QEasingCurve, QEventLoop, QPointF, QRectF, QSize, Qt, QVariantAnimation, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPainterPath, QPen, QRadialGradient
from PySide6.QtWidgets import QSizePolicy, QWidget

from .pieces import PieceSet
from .rules import WHITE, Move, Position, file_of, rank_of, square
from .theme import (BOARD_THEMES, DEFAULT_BOARD_THEME, ENGINE_ARROW, HOVER, LAST_MOVE, PREMOVE, SELECTED,
                    SHAPE_COLORS, TARGET, BoardTheme)

PROMOTION_ORDER = "qnrb"


def _rgba(values: tuple) -> QColor:
    return QColor(*values)


@dataclass
class AnimationPlan:
    """How to get from one position to the next on screen: pieces sliding
    (as drawn during the slide, from, to), pieces fading out (from the old
    position) and in (from the new)."""

    slides: list[tuple[str, int, int]] = field(default_factory=list)
    fade_out: list[tuple[str, int]] = field(default_factory=list)
    fade_in: list[tuple[str, int]] = field(default_factory=list)

    def hidden(self) -> set[int]:
        """Squares whose pieces of the new position wait until the end."""
        return {to for _p, _f, to in self.slides} | {sq for _p, sq in self.fade_in}


def plan_animation(before: Position, after: Position) -> AnimationPlan | None:
    """The motion between two positions one move apart (in either
    direction); None when nothing moves or too much changed to animate."""
    changed = [sq for sq in range(64) if before.board[sq] != after.board[sq]]
    if not 2 <= len(changed) <= 4:
        return None
    vanished = {sq: before.board[sq] for sq in changed if before.board[sq] != "."}
    appeared = {sq: after.board[sq] for sq in changed if after.board[sq] != "."}
    plan = AnimationPlan()

    def distance(a: int, b: int) -> int:
        return max(abs(file_of(a) - file_of(b)), abs(rank_of(a) - rank_of(b)))

    def match(same_piece: bool) -> None:
        for to, piece in sorted(appeared.items()):
            candidates = [sq for sq, p in vanished.items()
                          if (p == piece if same_piece else p.isupper() == piece.isupper())]
            if not candidates:
                continue
            frm = min(candidates, key=lambda sq: (distance(sq, to), sq))
            plan.slides.append((vanished.pop(frm), frm, to))
            del appeared[to]

    match(True)  # a piece that moved
    match(False)  # a pawn promoting (or the promoted piece going back): it slides as it was
    plan.fade_out = sorted((p, sq) for sq, p in vanished.items())
    plan.fade_in = sorted((p, sq) for sq, p in appeared.items())
    return plan if plan.slides else None


class BoardView(QWidget):
    move_made = Signal(object)  # a legal Move the user entered
    square_clicked = Signal(int, object)  # edit mode: square, mouse button
    resized = Signal()
    shapes_changed = Signal()  # the user drew or cleared arrows and circles
    premove_changed = Signal()  # a premove was queued or cancelled

    def __init__(self, piece_set: PieceSet, parent: QWidget | None = None):
        super().__init__(parent)
        self.position = Position.start()
        self.piece_set = piece_set
        self.board_theme: BoardTheme = BOARD_THEMES[DEFAULT_BOARD_THEME]
        self.flipped = False
        self.interactive = False  # may the user move pieces now?
        self.edit_mode = False  # report clicks instead of making moves
        self.show_coordinates = True
        self.show_legal = True  # dots on the squares the selected piece can go to
        self.highlight_last = True
        self.auto_queen = False  # promote to a queen without asking
        self.animation_ms = 160  # 0: no animation
        self.align_left = False  # board at the left edge instead of centered (next to an eval bar)
        self.last_move: Move | None = None
        self.arrows: list[Move] = []  # engine arrows (analysis)
        self.arrow_colors: list[tuple] = []  # their RGBA colors
        self.shape_arrows: dict[tuple[int, int], str] = {}  # (from, to) -> color name
        self.shape_circles: dict[int, str] = {}  # square -> color name
        self.premove_color: int | None = None  # the side that may queue a premove now
        self.premoves: list[Move] = []
        self._selected: int | None = None
        self._drag_from: int | None = None
        self._drag_pos: QPointF | None = None
        self._hover: int | None = None
        self._shape_start: tuple[int, str] | None = None
        self._plan: AnimationPlan | None = None
        self._plan_before: Position | None = None
        self._progress = 1.0
        self._animation = QVariantAnimation(self)
        self._animation.setStartValue(0.0)
        self._animation.setEndValue(1.0)
        self._animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._animation.valueChanged.connect(self._animate)
        self._animation.finished.connect(self._animation_done)
        self._promotion: tuple[int, int] | None = None  # (color, square) while the overlay asks
        self._promotion_loop: QEventLoop | None = None
        self._promotion_choice: str | None = None
        # (color, square) -> "q", "r", "b", "n" or None to cancel; replaceable (tests).
        self.promotion_chooser = self._ask_promotion
        self.setMinimumSize(200, 200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)

    def sizeHint(self) -> QSize:
        return QSize(560, 560)

    # --- State --------------------------------------------------------------------

    def set_position(self, position: Position, last_move: Move | None = None, animate: bool = True) -> None:
        """Shows `position`; a one-move change slides into place when
        animations are on and the board is on screen."""
        before = self.position
        self.position = position
        self.last_move = last_move
        self._cancel()
        self._stop_animation()
        if animate and self.animation_ms > 0 and self.isVisible():
            plan = plan_animation(before, position)
            if plan is not None:
                self._plan, self._plan_before, self._progress = plan, before, 0.0
                self._animation.setDuration(self.animation_ms)
                self._animation.start()
        self.update()

    def set_piece_set(self, piece_set: PieceSet) -> None:
        self.piece_set = piece_set
        self.update()

    def set_board_theme(self, theme: BoardTheme) -> None:
        self.board_theme = theme
        self.update()

    def set_flipped(self, flipped: bool) -> None:
        self.flipped = flipped
        self.update()

    def set_interactive(self, interactive: bool) -> None:
        self.interactive = interactive
        if not interactive and self.premove_color is None:
            self._cancel()
        self.update()

    def set_arrows(self, arrows: list[Move], colors: list[tuple] | None = None) -> None:
        """Engine arrows (analysis lines, hints, threats), each in its RGBA
        color (default: the engine arrow's)."""
        self.arrows = list(arrows)
        self.arrow_colors = list(colors) if colors is not None else [ENGINE_ARROW] * len(self.arrows)
        self.update()

    def set_shapes(self, arrows: dict[tuple[int, int], str], circles: dict[int, str]) -> None:
        self.shape_arrows, self.shape_circles = dict(arrows), dict(circles)
        self.update()

    def clear_shapes(self) -> None:
        if self.shape_arrows or self.shape_circles:
            self.shape_arrows, self.shape_circles = {}, {}
            self.shapes_changed.emit()
            self.update()

    def set_premove_color(self, color: int | None) -> None:
        """Lets `color` queue a premove (None: no premoves; the queued one stays
        until played or cancelled)."""
        self.premove_color = color
        if color is None and self._selected is not None and not self.interactive:
            self._cancel()
        self.update()

    def cancel_premoves(self) -> None:
        had = bool(self.premoves)
        self.premoves = []
        if self.premove_color is not None:
            self._cancel()
        if had:
            self.premove_changed.emit()
        self.update()

    def take_premove(self) -> Move | None:
        """The queued premove (removing it), or None."""
        if not self.premoves:
            return None
        move = self.premoves.pop(0)
        self.premove_changed.emit()
        self.update()
        return move

    def _cancel(self) -> None:
        self._selected = None
        self._drag_from = None
        self._drag_pos = None
        self._hover = None

    # --- Animation ------------------------------------------------------------------

    @property
    def animating(self) -> bool:
        return self._plan is not None

    def _animate(self, value) -> None:
        self._progress = float(value)
        self.update()

    def _animation_done(self) -> None:
        self._plan, self._plan_before, self._progress = None, None, 1.0
        self.update()

    def _stop_animation(self) -> None:
        if self._animation.state() == QVariantAnimation.State.Running:
            self._animation.stop()
        self._plan, self._plan_before, self._progress = None, None, 1.0

    def hideEvent(self, event) -> None:
        self._stop_animation()
        if self._promotion_loop is not None:
            self._promotion_loop.quit()
        super().hideEvent(event)

    # --- Geometry ------------------------------------------------------------------

    def board_rect(self) -> QRectF:
        side = math.floor(min(self.width(), self.height()) / 8) * 8
        x = 0 if self.align_left else (self.width() - side) / 2
        return QRectF(x, (self.height() - side) / 2, side, side)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.resized.emit()

    def square_size(self) -> float:
        return self.board_rect().width() / 8

    def square_rect(self, sq: int) -> QRectF:
        r, s = self.board_rect(), self.square_size()
        col = 7 - file_of(sq) if self.flipped else file_of(sq)
        row = rank_of(sq) if self.flipped else 7 - rank_of(sq)
        return QRectF(r.x() + col * s, r.y() + row * s, s, s)

    def square_at(self, point: QPointF) -> int | None:
        r, s = self.board_rect(), self.square_size()
        if s <= 0 or not r.contains(point):
            return None
        col = min(7, int((point.x() - r.x()) / s))
        row = min(7, int((point.y() - r.y()) / s))
        file, rank = (7 - col, row) if self.flipped else (col, 7 - row)
        return square(file, rank)

    def promotion_squares(self, color: int, to_sq: int) -> list[tuple[str, QRectF]]:
        """The overlay's choices (piece letter, rect): queen on the promotion
        square, then knight, rook and bishop toward the board's middle."""
        rect = self.square_rect(to_sq)
        down = rect.top() < self.board_rect().center().y()
        out = []
        for i, letter in enumerate(PROMOTION_ORDER):
            dy = rect.height() * i * (1 if down else -1)
            out.append((letter, rect.translated(0, dy)))
        return out

    # --- Painting -------------------------------------------------------------------

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        s = self.square_size()
        board = self.position.board
        light, dark = QColor(self.board_theme.light), QColor(self.board_theme.dark)
        for sq in range(64):
            p.fillRect(self.square_rect(sq), light if (file_of(sq) + rank_of(sq)) % 2 else dark)
        if self.highlight_last and self.last_move is not None:
            for sq in (self.last_move.from_sq, self.last_move.to_sq):
                p.fillRect(self.square_rect(sq), _rgba(LAST_MOVE))
        for m in self.premoves:
            for sq in (m.from_sq, m.to_sq):
                p.fillRect(self.square_rect(sq), _rgba(PREMOVE))
        if self._selected is not None:
            p.fillRect(self.square_rect(self._selected), _rgba(SELECTED))
        if not self.edit_mode and self._has_kings() and self.position.in_check():
            rect = self.square_rect(self.position.king_square(self.position.turn))
            glow = QRadialGradient(rect.center(), rect.width() * 0.6)
            glow.setColorAt(0.0, QColor(255, 0, 0, 230))
            glow.setColorAt(0.45, QColor(231, 0, 0, 160))
            glow.setColorAt(1.0, QColor(169, 0, 0, 0))
            p.fillRect(rect, QBrush(glow))
        if self._hover is not None and self._drag_from is not None:
            p.fillRect(self.square_rect(self._hover), _rgba(HOVER))
        if self.show_coordinates and s >= 24:
            self._paint_coordinates(p, s, light, dark)
        self._paint_pieces(p, s, board)
        if self._selected is not None and self.show_legal:
            self._paint_targets(p, s)
        for sq, color in self.shape_circles.items():
            self._paint_circle(p, sq, s, _rgba(SHAPE_COLORS.get(color, SHAPE_COLORS["green"])))
        for move, color in zip(self.arrows, self.arrow_colors):
            self._paint_arrow(p, move.from_sq, move.to_sq, s, _rgba(color))
        for (a, b), color in self.shape_arrows.items():
            self._paint_arrow(p, a, b, s, _rgba(SHAPE_COLORS.get(color, SHAPE_COLORS["green"])))
        ratio = self.devicePixelRatioF()
        size = max(1, int(s))
        if self._drag_from is not None and self._drag_pos is not None:
            piece = self.position.board[self._drag_from]
            if piece != ".":
                p.drawPixmap(QPointF(self._drag_pos.x() - size / 2, self._drag_pos.y() - size / 2),
                             self.piece_set.pixmap(piece, size, ratio))
        if self._promotion is not None:
            self._paint_promotion(p, s)
        p.end()

    def _piece_at(self, p: QPainter, piece: str, center: QPointF, size: int, opacity: float = 1.0) -> None:
        if opacity < 1.0:
            p.setOpacity(max(0.0, opacity))
        p.drawPixmap(QPointF(center.x() - size / 2, center.y() - size / 2),
                     self.piece_set.pixmap(piece, size, self.devicePixelRatioF()))
        if opacity < 1.0:
            p.setOpacity(1.0)

    def _paint_pieces(self, p: QPainter, s: float, board: str) -> None:
        size = max(1, int(s))
        plan = self._plan
        hidden = plan.hidden() if plan is not None else set()
        for sq in range(64):
            piece = board[sq]
            if piece == "." or sq in hidden or (sq == self._drag_from and self._drag_pos is not None):
                continue
            self._piece_at(p, piece, self.square_rect(sq).center(), size)
        if plan is None:
            return
        t = self._progress
        for piece, sq in plan.fade_out:
            self._piece_at(p, piece, self.square_rect(sq).center(), size, 1.0 - t)
        for piece, sq in plan.fade_in:
            self._piece_at(p, piece, self.square_rect(sq).center(), size, t)
        for piece, frm, to in plan.slides:
            a, b = self.square_rect(frm).center(), self.square_rect(to).center()
            self._piece_at(p, piece, QPointF(a.x() + (b.x() - a.x()) * t, a.y() + (b.y() - a.y()) * t), size)

    def _has_kings(self) -> bool:
        return self.position.board.count("K") == 1 and self.position.board.count("k") == 1

    def _paint_coordinates(self, p: QPainter, s: float, light: QColor, dark: QColor) -> None:
        font = QFont(self.font())
        font.setPixelSize(max(9, int(s * 0.17)))
        font.setBold(True)
        p.setFont(font)
        bottom_rank = 7 if self.flipped else 0
        left_file = 7 if self.flipped else 0
        for f in range(8):
            sq = square(f, bottom_rank)
            r = self.square_rect(sq).adjusted(0, 0, -s * 0.06, -s * 0.02)
            p.setPen(dark if (f + bottom_rank) % 2 else light)
            p.drawText(r, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom, "abcdefgh"[f])
        for rank in range(8):
            sq = square(left_file, rank)
            r = self.square_rect(sq).adjusted(s * 0.05, s * 0.03, 0, 0)
            p.setPen(dark if (left_file + rank) % 2 else light)
            p.drawText(r, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, str(rank + 1))

    def _paint_targets(self, p: QPainter, s: float) -> None:
        p.setPen(Qt.PenStyle.NoPen)
        color = _rgba(PREMOVE if self._premoving() else TARGET)
        for sq in self.targets(self._selected):
            rect = self.square_rect(sq)
            if self.position.board[sq] != "." or self.position.ep == sq:
                ring = QPainterPath()
                ring.addRect(rect)
                ring.addEllipse(rect.center(), s * 0.5, s * 0.5)
                p.fillPath(ring, color)
            else:
                p.setBrush(color)
                p.drawEllipse(rect.center(), s * 0.15, s * 0.15)

    def _paint_circle(self, p: QPainter, sq: int, s: float, color: QColor) -> None:
        pen = QPen(color, s * 0.07)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(self.square_rect(sq).center(), s * 0.46, s * 0.46)

    def _paint_arrow(self, p: QPainter, from_sq: int, to_sq: int, s: float, color: QColor) -> None:
        a, b = self.square_rect(from_sq).center(), self.square_rect(to_sq).center()
        dx, dy = b.x() - a.x(), b.y() - a.y()
        length = math.hypot(dx, dy)
        if length < 1:
            return
        ux, uy = dx / length, dy / length
        head, width = s * 0.42, s * 0.16
        tip = QPointF(b.x() - ux * s * 0.15, b.y() - uy * s * 0.15)
        neck = QPointF(tip.x() - ux * head, tip.y() - uy * head)
        start = QPointF(a.x() + ux * s * 0.2, a.y() + uy * s * 0.2)
        px, py = -uy, ux
        path = QPainterPath(QPointF(start.x() + px * width / 2, start.y() + py * width / 2))
        path.lineTo(neck.x() + px * width / 2, neck.y() + py * width / 2)
        path.lineTo(neck.x() + px * head * 0.55, neck.y() + py * head * 0.55)
        path.lineTo(tip)
        path.lineTo(neck.x() - px * head * 0.55, neck.y() - py * head * 0.55)
        path.lineTo(neck.x() - px * width / 2, neck.y() - py * width / 2)
        path.lineTo(start.x() - px * width / 2, start.y() - py * width / 2)
        path.closeSubpath()
        p.fillPath(path, color)

    def _paint_promotion(self, p: QPainter, s: float) -> None:
        color, to_sq = self._promotion
        p.fillRect(self.board_rect(), QColor(0, 0, 0, 120))
        size = max(1, int(s * 0.86))
        for letter, rect in self.promotion_squares(color, to_sq):
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(250, 250, 250, 235))
            p.drawEllipse(rect.center(), s * 0.48, s * 0.48)
            piece = letter.upper() if color == WHITE else letter
            self._piece_at(p, piece, rect.center(), size)

    # --- Input ---------------------------------------------------------------------

    def targets(self, sq: int) -> set[int]:
        if self._premoving():
            return self.premove_targets(sq)
        return {m.to_sq for m in self.position.legal_moves() if m.from_sq == sq}

    def premove_targets(self, sq: int) -> set[int]:
        """Where the piece on `sq` could go on its side's next turn, judged now
        (the opponent's coming move may change that)."""
        if self.premove_color is None:
            return set()
        pos = self.position
        mover = Position(pos.board, self.premove_color, pos.castling, None, pos.halfmove, pos.fullmove)
        return {m.to_sq for m in mover.pseudo_legal_moves() if m.from_sq == sq}

    def _premoving(self) -> bool:
        return not self.interactive and self.premove_color is not None and not self.edit_mode

    def _own(self, sq: int) -> bool:
        piece = self.position.board[sq]
        side = self.premove_color if self._premoving() else self.position.turn
        return piece != "." and piece.isupper() == (side == WHITE)

    @staticmethod
    def _shape_color(modifiers) -> str:
        if modifiers & Qt.KeyboardModifier.ShiftModifier:
            return "red"
        if modifiers & Qt.KeyboardModifier.ControlModifier:
            return "blue"
        if modifiers & Qt.KeyboardModifier.AltModifier:
            return "yellow"
        return "green"

    def mousePressEvent(self, event) -> None:
        sq = self.square_at(event.position())
        if self._promotion is not None:
            self._promotion_click(event.position())
            return
        if self.edit_mode:
            if sq is not None:
                self.square_clicked.emit(sq, event.button())
            return
        if event.button() == Qt.MouseButton.RightButton:
            had_premove = bool(self.premoves)
            self._cancel()
            if had_premove:
                self.cancel_premoves()
            self._shape_start = (sq, self._shape_color(event.modifiers())) if sq is not None else None
            self.update()
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        self.clear_shapes()
        if sq is None or not (self.interactive or self._premoving()):
            return
        if self._selected is not None and sq in self.targets(self._selected):
            self._try_move(self._selected, sq)
            return
        if self._own(sq) and self.targets(sq):
            self._selected = sq
            self._drag_from = sq
            self._drag_pos = event.position()
        else:
            self._selected = None
        self.update()

    def mouseMoveEvent(self, event) -> None:
        if self._drag_from is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self._drag_pos = event.position()
            sq = self.square_at(event.position())
            self._hover = sq if sq is not None and sq in self.targets(self._drag_from) else None
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.RightButton:
            start, self._shape_start = self._shape_start, None
            sq = self.square_at(event.position())
            if start is not None and sq is not None and not self.edit_mode:
                self._toggle_shape(start[0], sq, start[1])
            return
        if self._drag_from is None or event.button() != Qt.MouseButton.LeftButton:
            return
        src = self._drag_from
        self._drag_from = None
        self._drag_pos = None
        self._hover = None
        sq = self.square_at(event.position())
        if sq is not None and sq != src and sq in self.targets(src):
            self._try_move(src, sq)
        self.update()  # a release on the start square keeps it selected (click, then click the target)

    def _toggle_shape(self, a: int, b: int, color: str) -> None:
        if a == b:
            if self.shape_circles.get(a) == color:
                del self.shape_circles[a]
            else:
                self.shape_circles[a] = color
        elif self.shape_arrows.get((a, b)) == color:
            del self.shape_arrows[(a, b)]
        else:
            self.shape_arrows[(a, b)] = color
        self.shapes_changed.emit()
        self.update()

    def _try_move(self, from_sq: int, to_sq: int) -> None:
        if self._premoving():
            self._queue_premove(from_sq, to_sq)
            return
        candidates = [m for m in self.position.legal_moves() if m.from_sq == from_sq and m.to_sq == to_sq]
        if not candidates:
            return
        move = candidates[0]
        if len(candidates) > 1:  # a promotion: one move per piece
            choice = "q" if self.auto_queen else self.promotion_chooser(self.position.turn, to_sq)
            if choice not in ("q", "r", "b", "n"):
                self._cancel()
                self.update()
                return
            move = Move(from_sq, to_sq, choice)
        self._cancel()
        self.update()
        self.move_made.emit(move)

    def _queue_premove(self, from_sq: int, to_sq: int) -> None:
        piece = self.position.board[from_sq]
        last_rank = 7 if piece == "P" else 0
        promotion = "q" if piece in "Pp" and rank_of(to_sq) == last_rank else ""
        self.premoves = [Move(from_sq, to_sq, promotion)]
        self._cancel()
        self.premove_changed.emit()
        self.update()

    def _ask_promotion(self, color: int, to_sq: int) -> str | None:
        """The default chooser: an overlay on the board; a click outside it
        cancels. Waits in a local event loop, so the caller gets an answer."""
        self._promotion = (color, to_sq)
        self._promotion_choice = None
        self.update()
        loop = QEventLoop()
        self._promotion_loop = loop
        loop.exec()
        self._promotion_loop = None
        self._promotion = None
        self.update()
        return self._promotion_choice

    def _promotion_click(self, point: QPointF) -> None:
        color, to_sq = self._promotion
        self._promotion_choice = None
        for letter, rect in self.promotion_squares(color, to_sq):
            if rect.contains(point):
                self._promotion_choice = letter
        if self._promotion_loop is not None:
            self._promotion_loop.quit()
