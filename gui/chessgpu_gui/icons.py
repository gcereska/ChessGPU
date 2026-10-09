"""Toolbar and button icons, drawn with Qt paths in the theme's text color
(no image files): each name maps to a function that strokes or fills paths
in a 24 x 24 box."""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap


def _line(*points) -> QPainterPath:
    path = QPainterPath(QPointF(*points[0]))
    for p in points[1:]:
        path.lineTo(*p)
    return path


def _new(p: QPainter) -> None:
    p.drawPath(_line((12, 5), (12, 19)))
    p.drawPath(_line((5, 12), (19, 12)))


def _open(p: QPainter) -> None:
    p.drawPath(_line((3, 7), (3, 19), (19, 19), (21, 10), (7, 10), (4.5, 17)))
    p.drawPath(_line((3, 7), (3, 5), (9, 5), (11, 7), (18, 7), (18, 10)))


def _save(p: QPainter) -> None:
    p.drawPath(_line((4, 4), (17, 4), (20, 7), (20, 20), (4, 20), (4, 4)))
    p.drawPath(_line((8, 4), (8, 9), (16, 9), (16, 4)))
    p.drawRect(QRectF(8, 13, 8, 7))


def _flip(p: QPainter) -> None:
    p.drawPath(_line((8, 19), (8, 5)))
    p.drawPath(_line((4, 9), (8, 5), (12, 9)))
    p.drawPath(_line((16, 5), (16, 19)))
    p.drawPath(_line((12, 15), (16, 19), (20, 15)))


def _analyze(p: QPainter) -> None:
    p.drawEllipse(QPointF(10, 10), 6, 6)
    p.drawPath(_line((14.5, 14.5), (20, 20)))


def _settings(p: QPainter) -> None:
    p.drawEllipse(QPointF(12, 12), 3, 3)
    path = QPainterPath()
    teeth = 8
    for i in range(teeth * 2 + 1):
        a = math.pi * i / teeth
        r = 8.5 if i % 2 == 0 else 6.5
        for da in (-0.18, 0.18):
            x, y = 12 + r * math.cos(a + da), 12 + r * math.sin(a + da)
            if path.elementCount() == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)
    path.closeSubpath()
    p.drawPath(path)


def _chevrons(count: int, left: bool, bar: bool):
    def draw(p: QPainter) -> None:
        sign = -1 if left else 1
        start = 12 - sign * 2.5 * (count - 1) - (2 if bar else 0) * sign
        for i in range(count):
            x = start + sign * 5 * i
            p.drawPath(_line((x - sign * 3, 6), (x + sign * 3, 12), (x - sign * 3, 18)))
        if bar:
            x = 12 + sign * 6
            p.drawPath(_line((x, 6), (x, 18)))
    return draw


def _engine(p: QPainter) -> None:
    p.drawRect(QRectF(6, 6, 12, 12))
    p.drawRect(QRectF(9.5, 9.5, 5, 5))
    for k in (9, 12, 15):
        p.drawPath(_line((k, 3), (k, 6)))
        p.drawPath(_line((k, 18), (k, 21)))
        p.drawPath(_line((3, k), (6, k)))
        p.drawPath(_line((18, k), (21, k)))


def _human(p: QPainter) -> None:
    p.drawEllipse(QPointF(12, 8.5), 3.8, 3.8)
    p.drawPath(_line((5, 20), (5.5, 17), (8, 14.5), (16, 14.5), (18.5, 17), (19, 20)))


def _stop(p: QPainter) -> None:
    p.drawRect(QRectF(7, 7, 10, 10))


def _book(p: QPainter) -> None:
    p.drawPath(_line((12, 6), (12, 20)))
    p.drawPath(_line((12, 6), (9, 4.5), (3, 4.5), (3, 18.5), (9, 18.5), (12, 20)))
    p.drawPath(_line((12, 6), (15, 4.5), (21, 4.5), (21, 18.5), (15, 18.5), (12, 20)))


def _trophy(p: QPainter) -> None:
    p.drawPath(_line((7, 4), (17, 4), (16.5, 10), (14, 13.5), (10, 13.5), (7.5, 10), (7, 4)))
    p.drawPath(_line((7, 6), (4, 6), (4.5, 9), (7.5, 10)))
    p.drawPath(_line((17, 6), (20, 6), (19.5, 9), (16.5, 10)))
    p.drawPath(_line((12, 13.5), (12, 17)))
    p.drawPath(_line((8, 20), (16, 20), (15, 17), (9, 17), (8, 20)))


def _review(p: QPainter) -> None:
    p.drawPath(_line((3, 18), (8, 12), (12, 15), (17, 7), (21, 10)))
    p.drawPath(_line((3, 21), (21, 21)))


DRAWINGS = {
    "new": _new, "open": _open, "save": _save, "flip": _flip, "analyze": _analyze, "settings": _settings,
    "first": _chevrons(1, True, True), "previous": _chevrons(1, True, False),
    "next": _chevrons(1, False, False), "last": _chevrons(1, False, True),
    "engine": _engine, "human": _human, "stop": _stop, "book": _book, "trophy": _trophy, "review": _review,
}


def icon(name: str, color: str, size: int = 24) -> QIcon:
    """The icon `name` drawn in `color`, at 1x and 2x for sharp scaling."""
    out = QIcon()
    for scale in (1, 2):
        pm = QPixmap(size * scale, size * scale)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.scale(size * scale / 24, size * scale / 24)
        pen = QPen(QColor(color), 1.8)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        DRAWINGS[name](p)
        p.end()
        out.addPixmap(pm)
    return out
