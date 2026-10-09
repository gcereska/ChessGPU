"""The move tree: the main line in rows of two moves (as on lichess), with
comments and variations in rows between them; variations inside
variations in parentheses. Clicking a move shows its position; the context
menu annotates it (NAGs), promotes or deletes lines and copies them. A
comment box edits the shown move's comment."""

from __future__ import annotations

import html

from PySide6.QtCore import QEvent, Qt, QUrl, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QMenu, QPlainTextEdit, QTextBrowser, QToolTip, QWidget

from .game import GameRecord, Node
from .panels import figurine_san, format_clock
from .rules import WHITE
from .theme import current_theme

# Glyphs of the common NAGs: how good a move is (1-7), who stands better
# (10-19), and a few more (shown, not offered in the menu).
MOVE_NAGS = {1: "!", 2: "?", 3: "!!", 4: "??", 5: "!?", 6: "?!", 7: "□"}
POSITION_NAGS = {10: "=", 13: "∞", 14: "+=", 15: "=+", 16: "±", 17: "∓", 18: "+−", 19: "−+"}
OTHER_NAGS = {11: "=", 12: "=", 22: "zugzwang", 23: "zugzwang", 36: "↑", 37: "↑", 40: "→", 41: "→",
              132: "⇆", 133: "⇆", 138: "⊕", 139: "⊕", 140: "∆", 146: "N"}
NAG_TITLES = {1: "Good move", 2: "Mistake", 3: "Brilliant move", 4: "Blunder", 5: "Interesting move",
              6: "Dubious move", 7: "Only move", 10: "Equal position", 11: "Equal chances, quiet position",
              12: "Equal chances, active position", 13: "Unclear position", 14: "White is slightly better",
              15: "Black is slightly better", 16: "White is better", 17: "Black is better", 18: "White is winning",
              19: "Black is winning", 22: "White is in zugzwang", 23: "Black is in zugzwang",
              36: "White has the initiative", 37: "Black has the initiative", 40: "White has the attack",
              41: "Black has the attack", 132: "White has counterplay", 133: "Black has counterplay",
              138: "White is short of time", 139: "Black is short of time", 140: "With the idea", 146: "Novelty"}
NBSP = chr(0xA0)  # padding around a move (its highlight)
_NAG_COLORS = {3: "#3b9bd9", 5: "#b660cd", 6: "#5aa5d6"}  # the others come from the theme


def nag_color(nag: int) -> str:
    t = current_theme()
    return {1: t.success, 2: t.warning, 4: t.danger}.get(nag, _NAG_COLORS.get(nag, t.text_dim))


def nag_text(nags: list[int]) -> tuple[str, str]:
    """(move glyphs, position glyphs) for a node's NAGs; others as $n."""
    move = "".join(MOVE_NAGS[n] for n in nags if n in MOVE_NAGS)
    rest = " ".join(POSITION_NAGS.get(n) or OTHER_NAGS.get(n) or f"${n}" for n in nags if n not in MOVE_NAGS)
    return move, rest


def toggle_nag(nags: list[int], nag: int) -> list[int]:
    """The NAGs after choosing `nag`: chosen again it goes; otherwise it
    replaces another of its kind (a move's quality, or the position's)."""
    if nag in nags:
        return [n for n in nags if n != nag]
    group = MOVE_NAGS if nag in MOVE_NAGS else POSITION_NAGS if nag in POSITION_NAGS else {}
    return [n for n in nags if n not in group] + [nag]


def line_text(node: Node, figurines: bool = False) -> str:
    """The moves from the start to `node` as numbered SAN."""
    parts = []
    path = node.path()[1:]
    for i, n in enumerate(path):
        before = n.parent.position
        if before.turn == WHITE:
            parts.append(f"{before.fullmove}.")
        elif i == 0:
            parts.append(f"{before.fullmove}...")
        parts.append(figurine_san(n.san) if figurines else n.san)
    return " ".join(parts)


class MoveTreeView(QTextBrowser):
    """The game's moves, variations and comments; the shown move highlighted."""

    node_selected = Signal(object)  # Node
    action_requested = Signal(str, object)  # action name, Node: see _menu

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("moveTree")
        self.setOpenLinks(False)
        self.setOpenExternalLinks(False)
        self.anchorClicked.connect(self._clicked)
        self.figurines = False
        self.can_edit = True  # promote and delete (not while a game runs)
        self.record: GameRecord | None = None
        self.current: Node | None = None
        self._nodes: list[Node] = []  # anchor "n:i" -> self._nodes[i]

    # --- Content ---------------------------------------------------------------------

    def set_tree(self, record: GameRecord, current: Node) -> None:
        self.record, self.current = record, current
        self.refresh()

    def refresh(self) -> None:
        if self.record is None:
            return
        scroll = self.verticalScrollBar().value()
        self.setHtml(self._html())
        self.verticalScrollBar().setValue(scroll)
        self._show_current()

    def anchors(self) -> list[tuple[Node, str]]:
        """(node, text shown) for every move in the view, in order."""
        order: list[int] = []
        texts: dict[int, str] = {}
        block = self.document().begin()
        while block.isValid():
            it = block.begin()
            while not it.atEnd():
                frag = it.fragment()
                i = self._index(frag.charFormat().anchorHref()) if frag.isValid() else None
                if i is not None:
                    if i not in texts:
                        order.append(i)
                        texts[i] = ""
                    texts[i] += frag.text()  # (a move's glyphs are fragments of their own)
                it += 1
            block = block.next()
        return [(self._nodes[i], texts[i].replace(NBSP, " ").strip()) for i in order]

    def node_at(self, point) -> Node | None:
        i = self._index(self.anchorAt(point))
        return self._nodes[i] if i is not None else None

    def _index(self, href: str) -> int | None:
        if not href.startswith("n:"):
            return None
        try:
            i = int(href[2:])
        except ValueError:
            return None
        return i if 0 <= i < len(self._nodes) else None

    def _show_current(self) -> None:
        """Scrolls (as little as needed) to the highlighted move."""
        if self.current is None or self.current.is_root:
            return
        href = next((f"n:{i}" for i, n in enumerate(self._nodes) if n is self.current), None)
        block = self.document().begin()
        while block.isValid():
            it = block.begin()
            while not it.atEnd():
                frag = it.fragment()
                if frag.isValid() and frag.charFormat().anchorHref() == href:
                    cursor = self.textCursor()
                    cursor.setPosition(frag.position())
                    self.setTextCursor(cursor)
                    self.ensureCursorVisible()
                    return
                it += 1
            block = block.next()

    # --- HTML ----------------------------------------------------------------------------

    def _html(self) -> str:
        t = current_theme()
        r = self.record
        self._nodes = []
        rows = []
        if r.root.comment:
            rows.append(self._wide_row(self._comment(r.root.comment)))
        stripe = [0]

        def move_row(number, white, black):
            stripe[0] += 1
            color = t.surface_alt if stripe[0] % 2 == 0 else t.surface
            rows.append(f'<tr bgcolor="{color}"><td width="38" style="color:{t.text_dim}">{number}.</td>'
                        f'<td width="32%">{white}</td><td>{black}</td></tr>')

        pending = None  # (number, White's cell) waiting for Black's move
        for node in r.moves:
            before = node.parent.position
            cell = self._anchor(node, main=True)
            if before.turn == WHITE:
                pending = (before.fullmove, cell)
            else:
                move_row(*(pending or (before.fullmove, f'<span style="color:{t.text_dim}">…</span>')), cell)
                pending = None
            variations = node.parent.children[1:]
            if node.comment or variations:
                if pending:
                    move_row(*pending, "")
                    pending = None
                if node.comment:
                    rows.append(self._wide_row(self._comment(node.comment)))
                for v in variations:
                    rows.append(self._wide_row(self._variation(v), variation=True))
        if pending:
            move_row(*pending, "")
        if r.result != "*" or r.reason:
            text = html.escape(" · ".join(p for p in (r.result if r.result != "*" else "", r.reason) if p))
            rows.append(f'<tr><td colspan="3" align="center" style="color:{t.text_dim}; padding:6px">'
                        f'<b>{text}</b></td></tr>')
        if not r.moves and not r.root.comment:
            rows.append(f'<tr><td colspan="3" style="color:{t.text_dim}; padding:8px">'
                        f'<i>No moves yet.</i></td></tr>')
        return (f'<html><body style="color:{t.text}">'
                f'<table width="100%" cellspacing="0" cellpadding="3" border="0">{"".join(rows)}</table>'
                f'</body></html>')

    def _wide_row(self, content: str, variation: bool = False) -> str:
        margin = 14 if variation else 6
        return (f'<tr><td colspan="3"><p style="margin-left:{margin}px; margin-top:1px; margin-bottom:1px">'
                f'{content}</p></td></tr>')

    def _comment(self, text: str) -> str:
        return f'<span style="color:{current_theme().text_dim}"><i>{html.escape(text)}</i></span>'

    def _anchor(self, node: Node, main: bool) -> str:
        """A move: a link "n:i", highlighted if shown, with its NAG glyphs."""
        t = current_theme()
        i = len(self._nodes)
        self._nodes.append(node)
        san = html.escape(figurine_san(node.san) if self.figurines else node.san)
        move_glyphs, position_glyphs = nag_text(node.nags)
        style = "text-decoration:none;" + ("font-weight:600;" if main else "")
        if node is self.current:
            style += f"color:{t.accent_text}; background-color:{t.accent};"
            glyph = html.escape(move_glyphs)
        else:
            style += f"color:{t.text if main else t.text_dim};"
            first = next((n for n in node.nags if n in MOVE_NAGS), None)
            glyph = (f'<span style="color:{nag_color(first)}">{html.escape(move_glyphs)}</span>'
                     if move_glyphs else "")
        pad = "&nbsp;" if main else ""
        out = f'<a href="n:{i}" style="{style}">{pad}{san}{glyph}{pad}</a>'
        if position_glyphs:
            out += f' <span style="color:{t.text_dim}">{html.escape(position_glyphs)}</span>'
        return out

    def _numbered(self, node: Node, number: bool) -> str:
        before = node.parent.position
        if before.turn == WHITE:
            prefix = f"{before.fullmove}."
        elif number:
            prefix = f"{before.fullmove}..."
        else:
            return self._anchor(node, main=False)
        return f'<span style="color:{current_theme().text_dim}">{prefix}</span> {self._anchor(node, main=False)}'

    def _variation(self, first: Node) -> str:
        """A variation: `first` and the line it continues in, comments and
        the variations within it (in parentheses)."""
        parts = []
        if first.starting_comment:
            parts.append(self._comment(first.starting_comment))
        node, number = first, True
        while True:
            parts.append(self._numbered(node, number))
            number = False
            if node.comment:
                parts.append(self._comment(node.comment))
                number = True
            if node is not first:
                for v in node.parent.children[1:]:
                    parts.append(f"({self._variation(v)})")
                    number = True
            if not node.children:
                break
            node = node.children[0]
        return " ".join(parts)

    # --- Mouse -----------------------------------------------------------------------------

    def _clicked(self, url: QUrl) -> None:
        i = self._index(url.toString())
        if i is not None:
            self.node_selected.emit(self._nodes[i])

    def viewportEvent(self, event) -> bool:
        if event.type() == QEvent.Type.ToolTip:
            node = self.node_at(event.pos())
            text = self.tooltip(node) if node is not None else ""
            if text:
                QToolTip.showText(event.globalPos(), text, self.viewport())
            else:
                QToolTip.hideText()
            return True
        return super().viewportEvent(event)

    @staticmethod
    def tooltip(node: Node) -> str:
        """What a move's tooltip shows: NAG titles, the engine's report, the
        review's evaluation, the clock and notes."""
        parts = [NAG_TITLES[n] for n in node.nags if n in NAG_TITLES]
        if node.eval is not None and (node.eval.score is not None or node.eval.seconds is not None):
            parts.append(node.eval.comment())
        if node.review is not None:
            parts.append(f"review: {node.review.score.text()}" +
                         (f" (depth {node.review.depth})" if node.review.depth else ""))
        if node.clock_ms is not None:
            parts.append(f"clock {format_clock(node.clock_ms)}")
        if node.note:
            parts.append(node.note)
        return "\n".join(parts)

    def contextMenuEvent(self, event) -> None:
        node = self.node_at(event.pos())
        if node is None:
            super().contextMenuEvent(event)
            return
        self._menu(node).exec(event.globalPos())

    def _menu(self, node: Node) -> QMenu:
        """The context menu for a move (kept as self.menu for tests)."""
        menu = QMenu(self)

        def ask(name):
            return lambda _checked=False: self.action_requested.emit(name, node)

        annotate = menu.addMenu("Annotate")
        for table in (MOVE_NAGS, POSITION_NAGS):
            for nag, glyph in table.items():
                a = QAction(f"{glyph}\t{NAG_TITLES[nag]}", annotate)
                a.setCheckable(True)
                a.setChecked(nag in node.nags)
                a.triggered.connect(ask(f"nag:{nag}"))
                annotate.addAction(a)
            annotate.addSeparator()
        clear = annotate.addAction("Clear annotations")
        clear.setEnabled(bool(node.nags))
        clear.triggered.connect(ask("clear_nags"))
        menu.addAction("Edit comment").triggered.connect(ask("comment"))
        menu.addSeparator()
        main = node.is_main()
        for name, text, enabled in (("promote", "Promote variation", not main and self.can_edit),
                                    ("make_main", "Make main line", not main and self.can_edit),
                                    ("delete", "Delete from here", self.can_edit)):
            a = menu.addAction(text)
            a.setEnabled(enabled)
            a.triggered.connect(ask(name))
        menu.addSeparator()
        menu.addAction("Copy line").triggered.connect(ask("copy_line"))
        menu.addAction("Copy FEN").triggered.connect(ask("copy_fen"))
        self.menu = menu
        return menu


class CommentBox(QPlainTextEdit):
    """Edits the shown move's comment (the game's, at the start)."""

    edited = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("commentBox")
        self.setMaximumHeight(64)
        self.setTabChangesFocus(True)
        self.node: Node | None = None
        self.textChanged.connect(self._changed)

    def set_node(self, node: Node) -> None:
        if node is self.node:
            return  # (the text may be mid-edit)
        self.node = node
        self.blockSignals(True)
        self.setPlainText(node.comment)
        self.blockSignals(False)
        self.setPlaceholderText("Comment on the game (before the first move)" if node.is_root else
                                f"Comment on {node.san}")

    def _changed(self) -> None:
        if self.node is not None:
            self.node.comment = self.toPlainText().strip()
            self.edited.emit()
