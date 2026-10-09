"""Piece sets: three ways to draw the pieces, chosen in the View menu.

- cburnett: the SVG set by Cburnett (Wikimedia Commons, BSD 3-clause; see
  assets/pieces/cburnett/LICENSE.txt), the look of Wikipedia and lichess.
- glyphs: the Unicode chess symbols of a system font, drawn as outlines and
  filled, so both colors look alike; a letter in a disc if no font has them.
- vector: our own drawings, made of Qt paths.

Every set paints one piece (a FEN letter) into a rectangle; pixmap() caches
the result per size and device pixel ratio.
"""

from __future__ import annotations

import pathlib

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QFontDatabase, QFontMetrics, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtSvg import QSvgRenderer

ASSETS = pathlib.Path(__file__).resolve().parent / "assets" / "pieces"
PIECES = "KQRBNPkqrbnp"
WHITE_FILL, WHITE_LINE = QColor("#fafafa"), QColor("#101010")
BLACK_FILL, BLACK_LINE, BLACK_DETAIL = QColor("#1f1f1f"), QColor("#000000"), QColor("#e6e6e6")


class PieceSet:
    name = ""
    title = ""

    def __init__(self):
        self._cache: dict[tuple, QPixmap] = {}

    def paint(self, painter: QPainter, rect: QRectF, piece: str) -> None:
        raise NotImplementedError

    def pixmap(self, piece: str, size: int, ratio: float = 1.0) -> QPixmap:
        key = (piece, size, ratio)
        pm = self._cache.get(key)
        if pm is None:
            px = max(1, round(size * ratio))
            pm = QPixmap(px, px)
            pm.fill(Qt.GlobalColor.transparent)
            painter = QPainter(pm)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            self.paint(painter, QRectF(0, 0, px, px), piece)
            painter.end()
            pm.setDevicePixelRatio(ratio)
            self._cache[key] = pm
        return pm


class SvgPieceSet(PieceSet):
    name = "cburnett"
    title = "cburnett (SVG)"

    def __init__(self, folder: pathlib.Path = ASSETS / "cburnett"):
        super().__init__()
        self._renderers = {}
        for p in PIECES:
            path = folder / f"Chess_{p.lower()}{'l' if p.isupper() else 'd'}t45.svg"
            renderer = QSvgRenderer(str(path))
            if not renderer.isValid():
                raise FileNotFoundError(f"bad or missing piece image {path}")
            self._renderers[p] = renderer

    def paint(self, painter: QPainter, rect: QRectF, piece: str) -> None:
        self._renderers[piece].render(painter, rect)


# Glyph code points: solid ("black") and outline ("white") chess symbols.
_SOLID = {"K": 0x265A, "Q": 0x265B, "R": 0x265C, "B": 0x265D, "N": 0x265E, "P": 0x265F}
_OUTLINE = {"K": 0x2654, "Q": 0x2655, "R": 0x2656, "B": 0x2657, "N": 0x2658, "P": 0x2659}
_GLYPH_FONTS = ("Segoe UI Symbol", "DejaVu Sans", "Noto Sans Symbols2", "Arial Unicode MS", "FreeSerif", "Symbola")


def _glyph_font() -> QFont | None:
    """An installed font that has every chess symbol, or None."""
    installed = set(QFontDatabase.families())
    for family in _GLYPH_FONTS:
        if family not in installed:
            continue
        font = QFont(family)
        font.setPixelSize(100)
        metrics = QFontMetrics(font)
        if all(metrics.inFontUcs4(c) for c in list(_SOLID.values()) + list(_OUTLINE.values())):
            return font
    return None


def _silhouette(path: QPainterPath) -> QPainterPath:
    """The path with its holes filled: the union of every contour as a solid."""
    out = QPainterPath()
    for polygon in path.toSubpathPolygons():
        piece = QPainterPath()
        piece.addPolygon(polygon)
        piece.closeSubpath()
        out = out.united(piece)
    return out


class GlyphPieceSet(PieceSet):
    """White pieces: the solid glyph's silhouette in white under the outline
    glyph in black. Black pieces: the solid glyph in black over a light
    silhouette, so its inner details show as light lines."""

    name = "glyphs"
    title = "Unicode glyphs"

    def __init__(self, font: QFont | None = None, use_font: bool = True):
        super().__init__()
        self.font = (font or _glyph_font()) if use_font else None
        self._shapes: dict[str, tuple] = {}

    @property
    def has_font(self) -> bool:
        return self.font is not None

    def _text(self, code: int) -> QPainterPath:
        path = QPainterPath()
        path.addText(0, 0, self.font, chr(code))
        return path

    def _shapes_for(self, kind: str) -> tuple:
        """(solid, silhouette, outline); without a font, a disc with the letter cut out."""
        shapes = self._shapes.get(kind)
        if shapes is None:
            solid = self._text(_SOLID[kind]) if self.font is not None else QPainterPath()
            if not solid.isEmpty():
                shapes = (solid, _silhouette(solid), self._text(_OUTLINE[kind]))
            else:
                disc = QPainterPath()
                disc.addEllipse(QRectF(-45, -45, 90, 90))
                font = QFont()
                font.setPixelSize(60)
                font.setBold(True)
                letter = QPainterPath()
                letter.addText(0, 0, font, kind)
                letter.translate(-letter.boundingRect().center())
                shapes = (disc.subtracted(letter), None, None)
            self._shapes[kind] = shapes
        return shapes

    def paint(self, painter: QPainter, rect: QRectF, piece: str) -> None:
        solid, silhouette, outline = self._shapes_for(piece.upper())
        box = solid.boundingRect()
        if box.isEmpty():
            return
        scale = 0.86 * min(rect.width() / box.width(), rect.height() / box.height())
        white = piece.isupper()
        painter.save()
        painter.translate(rect.center())
        painter.scale(scale, scale)
        painter.translate(-box.center())
        hairline = QPen(BLACK_LINE, 1.0 / scale)
        if silhouette is None:  # the letter disc
            painter.setPen(QPen(WHITE_LINE if white else BLACK_LINE, 3.0))
            painter.setBrush(QBrush(WHITE_FILL if white else BLACK_FILL))
            painter.drawPath(solid)
        elif white:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(WHITE_FILL))
            painter.drawPath(silhouette)
            painter.setBrush(QBrush(WHITE_LINE))
            painter.drawPath(outline)
        else:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(BLACK_DETAIL))
            painter.drawPath(silhouette)
            painter.setPen(hairline)
            painter.setBrush(QBrush(BLACK_FILL))
            painter.drawPath(solid)
        painter.restore()


def _shape(build) -> QPainterPath:
    path = QPainterPath()
    build(path)
    return path


def _rect(x, y, w, h, r=0.0) -> QPainterPath:
    return _shape(lambda p: p.addRoundedRect(QRectF(x, y, w, h), r, r) if r else p.addRect(QRectF(x, y, w, h)))


def _circle(x, y, r) -> QPainterPath:
    return _shape(lambda p: p.addEllipse(QPointF(x, y), r, r))


def _union(*paths: QPainterPath) -> QPainterPath:
    """One outline around shapes that overlap (adding them to one path
    instead would draw their inner edges, and the odd-even fill rule would
    punch holes where they overlap)."""
    out = QPainterPath()
    for path in paths:
        out = out.united(path)
    return out.simplified()


def _curve(start, *segments, close=True) -> QPainterPath:
    """A path from `start` through segments: (x, y) lines or 6-number cubics."""
    path = QPainterPath(QPointF(*start))
    for s in segments:
        if len(s) == 2:
            path.lineTo(*s)
        else:
            path.cubicTo(*s)
    if close:
        path.closeSubpath()
    return path


def _vector_paths(kind: str) -> tuple[QPainterPath, QPainterPath, QPainterPath]:
    """Our own drawing of a piece in a 100 x 100 box: (body, lines, dots).
    Lines and dots are details in a contrasting color."""
    base = _rect(24, 80, 52, 10, 4)
    lines, dots = QPainterPath(), QPainterPath()
    if kind == "P":
        stem = _curve((35, 80), (36, 66, 42, 56, 44, 48), (56, 48), (58, 56, 64, 66, 65, 80))
        body = _union(base, stem, _rect(38, 41, 24, 7, 3), _circle(50, 30, 11))
    elif kind == "R":
        tower = _curve((34, 72), (37, 38), (63, 38), (66, 72))
        merlons = [_rect(x, 14, 9, 18) for x in (30, 45.5, 61)]
        body = _union(base, _rect(30, 72, 40, 8), tower, _rect(30, 31, 40, 7), _rect(30, 22, 40, 10), *merlons)
        lines = _curve((37, 66), (63, 66), close=False)
    elif kind == "B":
        mitre = _curve((36, 72), (30, 56, 38, 36, 50, 24), (62, 36, 70, 56, 64, 72))
        body = _union(base, mitre, _rect(34, 66, 32, 14, 4), _circle(50, 17, 6))
        lines = _curve((57, 39), (45, 53), close=False)
        lines.addPath(_curve((36, 72), (64, 72), close=False))
    elif kind == "N":
        head = _curve((70, 80), (74, 60, 76, 40, 62, 24), (60, 11), (52, 19), (42, 21, 30, 31, 26, 41), (19, 52),
                      (21, 59, 30, 59, 36, 55), (44, 51, 48, 53, 46, 59), (42, 67, 36, 72, 34, 80))
        body = _union(base, head)
        lines = _curve((62, 26), (70, 40, 70, 58, 66, 74), close=False)
        dots = _circle(45, 32, 2.6)
    elif kind == "Q":
        crown = _curve((30, 80), (22, 34), (36, 56), (36, 26), (46, 54), (50, 22), (54, 54), (64, 26), (64, 56),
                       (78, 34), (70, 80))
        balls = [_circle(x, y, 4.5) for x, y in ((22, 30), (36, 22), (50, 18), (64, 22), (78, 30))]
        body = _union(base, crown, *balls)
        lines = _curve((31, 70), (69, 70), close=False)
    elif kind == "K":
        lobes = _curve((30, 80), (22, 62, 24, 44, 36, 40), (42, 38, 46, 42, 50, 46), (54, 42, 58, 38, 64, 40),
                       (76, 44, 78, 62, 70, 80))
        body = _union(base, lobes, _rect(46, 8, 8, 34), _rect(38, 15, 24, 8))
        lines = _curve((31, 70), (69, 70), close=False)
        lines.addPath(_curve((50, 47), (50, 66), close=False))
    else:
        raise ValueError(f"unknown piece kind {kind}")
    return body, lines, dots


class VectorPieceSet(PieceSet):
    name = "vector"
    title = "Vector (our own)"

    def __init__(self):
        super().__init__()
        self._paths = {k: _vector_paths(k) for k in "KQRBNP"}

    def paint(self, painter: QPainter, rect: QRectF, piece: str) -> None:
        body, lines, dots = self._paths[piece.upper()]
        white = piece.isupper()
        painter.save()
        painter.translate(rect.topLeft())
        painter.scale(rect.width() / 100, rect.height() / 100)
        pen = QPen(WHITE_LINE if white else BLACK_LINE, 3.0)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(QBrush(WHITE_FILL if white else BLACK_FILL))
        painter.drawPath(body)
        detail = WHITE_LINE if white else BLACK_DETAIL
        line_pen = QPen(detail, 2.6)
        line_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(line_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(lines)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(detail))
        painter.drawPath(dots)
        painter.restore()


PIECE_SETS = {cls.name: cls for cls in (SvgPieceSet, GlyphPieceSet, VectorPieceSet)}
DEFAULT_PIECE_SET = "cburnett"


def make_piece_set(name: str) -> PieceSet:
    return PIECE_SETS.get(name, PIECE_SETS[DEFAULT_PIECE_SET])()
