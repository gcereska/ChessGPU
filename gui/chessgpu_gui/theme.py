"""Themes: the application's colors, dark and light, as named tokens; a Qt
style sheet and palette built from them; and board color themes.

A Theme is a set of colors with fixed roles (window, surfaces, text, accent,
states). style_sheet() turns one into the application's style sheet, so
every widget follows the theme without colors of its own; widgets that paint
themselves (board, eval bar, graph, player cards) read the current theme
from current_theme().
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, fields

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication


@dataclass(frozen=True)
class Theme:
    name: str
    title: str
    window: str  # behind everything
    surface: str  # panels, lists, inputs
    surface_alt: str  # alternate rows, headers
    raised: str  # buttons, cards
    border: str
    text: str
    text_dim: str  # secondary text: labels, hints
    accent: str  # selection, focus, the current move
    accent_text: str  # text on the accent color
    hover: str
    danger: str  # low time, errors
    warning: str
    success: str  # a running clock
    clock: str  # a clock at rest
    clock_active: str  # the running clock
    clock_low: str  # the running clock under 20 seconds
    graph: str  # the eval graph's background
    graph_fill: str  # White's area in the eval graph
    eval_white: str
    eval_black: str


THEMES: dict[str, Theme] = {
    "dark": Theme(
        name="dark", title="Dark",
        window="#1e1f22", surface="#2b2d31", surface_alt="#313338", raised="#383a40", border="#45474e",
        text="#e3e5e8", text_dim="#a2a6ad", accent="#4a7fd4", accent_text="#ffffff", hover="#3a3c42",
        danger="#e5534b", warning="#d29922", success="#57ab5a",
        clock="#2b2d31", clock_active="#2f4a33", clock_low="#5c2b29",
        graph="#26272b", graph_fill="#c9ccd1", eval_white="#f2f2f2", eval_black="#3a3b40"),
    "light": Theme(
        name="light", title="Light",
        window="#eef0f3", surface="#ffffff", surface_alt="#f5f6f8", raised="#ffffff", border="#d0d4da",
        text="#1f2328", text_dim="#5b6570", accent="#2f6fdf", accent_text="#ffffff", hover="#e6e9ee",
        danger="#cf222e", warning="#9a6700", success="#1a7f37",
        clock="#ffffff", clock_active="#dcefe0", clock_low="#fbe0df",
        graph="#f5f6f8", graph_fill="#3d4148", eval_white="#fbfbfb", eval_black="#41454c"),
}
DEFAULT_THEME = "dark"

# The tick drawn in checked checkboxes (style sheets draw the indicators themselves:
# with the window background rule, Fusion's own indicator frames don't show).
CHECK_MARK = pathlib.Path(__file__).resolve().parent / "assets" / "ui" / "check.svg"

_current: Theme = THEMES[DEFAULT_THEME]


def current_theme() -> Theme:
    return _current


def theme_colors(theme: Theme) -> dict[str, str]:
    """The color tokens of a theme (every field but name and title)."""
    return {f.name: getattr(theme, f.name) for f in fields(theme) if f.name not in ("name", "title")}


def style_sheet(t: Theme) -> str:
    """The application's style sheet for theme `t`."""
    return f"""
QWidget {{ background-color: {t.window}; color: {t.text}; }}
QMainWindow, QDialog {{ background-color: {t.window}; }}
QToolTip {{ background-color: {t.raised}; color: {t.text}; border: 1px solid {t.border}; padding: 4px; }}
QMenuBar {{ background-color: {t.window}; border-bottom: 1px solid {t.border}; }}
QMenuBar::item {{ padding: 5px 10px; background: transparent; }}
QMenuBar::item:selected {{ background-color: {t.hover}; border-radius: 4px; }}
QMenu {{ background-color: {t.surface}; border: 1px solid {t.border}; padding: 4px; }}
QMenu::item {{ padding: 5px 24px 5px 22px; border-radius: 4px; }}
QMenu::item:selected {{ background-color: {t.accent}; color: {t.accent_text}; }}
QMenu::item:disabled {{ color: {t.text_dim}; }}
QMenu::separator {{ height: 1px; background: {t.border}; margin: 4px 6px; }}
QToolBar {{ background-color: {t.window}; border: none; border-bottom: 1px solid {t.border}; padding: 3px; spacing: 2px; }}
QToolButton {{ background: transparent; border: 1px solid transparent; border-radius: 5px; padding: 4px 8px; }}
QToolButton:hover {{ background-color: {t.hover}; border-color: {t.border}; }}
QToolButton:checked {{ background-color: {t.accent}; color: {t.accent_text}; }}
QToolButton:disabled {{ color: {t.text_dim}; }}
QPushButton {{ background-color: {t.raised}; border: 1px solid {t.border}; border-radius: 5px; padding: 5px 14px; }}
QPushButton:hover {{ background-color: {t.hover}; }}
QPushButton:pressed, QPushButton:checked {{ background-color: {t.accent}; color: {t.accent_text}; }}
QPushButton:default {{ border-color: {t.accent}; }}
QPushButton:disabled {{ color: {t.text_dim}; }}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTextEdit, QTextBrowser {{
    background-color: {t.surface}; border: 1px solid {t.border}; border-radius: 5px; padding: 3px 6px;
    selection-background-color: {t.accent}; selection-color: {t.accent_text}; }}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QPlainTextEdit:focus {{ border-color: {t.accent}; }}
QComboBox QAbstractItemView {{ background-color: {t.surface}; border: 1px solid {t.border};
    selection-background-color: {t.accent}; selection-color: {t.accent_text}; }}
QCheckBox, QRadioButton, QLabel {{ background: transparent; }}
QCheckBox::indicator, QRadioButton::indicator {{ width: 14px; height: 14px; border: 1px solid {t.text_dim};
    background-color: {t.surface}; }}
QCheckBox::indicator {{ border-radius: 3px; }}
QRadioButton::indicator {{ border-radius: 8px; }}
QCheckBox::indicator:hover, QRadioButton::indicator:hover {{ border-color: {t.accent}; }}
QCheckBox::indicator:checked {{ background-color: {t.accent}; border-color: {t.accent};
    image: url("{CHECK_MARK.as_posix()}"); }}
QRadioButton::indicator:checked {{ border-color: {t.accent}; background-color: qradialgradient(cx: 0.5, cy: 0.5,
    radius: 0.5, fx: 0.5, fy: 0.5, stop: 0 {t.accent}, stop: 0.45 {t.accent}, stop: 0.55 {t.surface}, stop: 1 {t.surface}); }}
QCheckBox::indicator:disabled, QRadioButton::indicator:disabled {{ border-color: {t.border}; background-color: {t.surface_alt}; }}
QCheckBox::indicator:checked:disabled {{ background-color: {t.border}; }}
QGroupBox {{ border: 1px solid {t.border}; border-radius: 6px; margin-top: 12px; padding-top: 6px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {t.text_dim}; }}
QTableWidget, QTableView, QListWidget, QListView, QTreeView, QTreeWidget {{
    background-color: {t.surface}; alternate-background-color: {t.surface_alt}; border: 1px solid {t.border};
    border-radius: 6px; gridline-color: transparent; selection-background-color: {t.accent};
    selection-color: {t.accent_text}; }}
QHeaderView::section {{ background-color: {t.surface_alt}; color: {t.text_dim}; border: none;
    border-bottom: 1px solid {t.border}; padding: 4px 6px; }}
QTableCornerButton::section {{ background-color: {t.surface_alt}; border: none; }}
QTabWidget::pane {{ border: 1px solid {t.border}; border-radius: 6px; top: -1px; background-color: {t.window}; }}
QTabBar::tab {{ background: transparent; color: {t.text_dim}; padding: 6px 14px; border: none;
    border-bottom: 2px solid transparent; }}
QTabBar::tab:selected {{ color: {t.text}; border-bottom: 2px solid {t.accent}; }}
QTabBar::tab:hover {{ color: {t.text}; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle {{ background-color: {t.border}; border-radius: 4px; min-height: 24px; min-width: 24px; }}
QScrollBar::handle:hover {{ background-color: {t.text_dim}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
QSplitter::handle {{ background-color: {t.window}; }}
QSplitter::handle:hover {{ background-color: {t.border}; }}
QDockWidget {{ titlebar-close-icon: none; }}
QDockWidget::title {{ background-color: {t.surface_alt}; padding: 4px 8px; }}
QStatusBar {{ background-color: {t.window}; color: {t.text_dim}; border-top: 1px solid {t.border}; }}
QStatusBar QLabel {{ color: {t.text_dim}; }}
QProgressBar {{ background-color: {t.surface}; border: 1px solid {t.border}; border-radius: 5px; text-align: center; }}
QProgressBar::chunk {{ background-color: {t.accent}; border-radius: 4px; }}
QSlider::groove:horizontal {{ height: 4px; background: {t.border}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {t.accent}; width: 14px; margin: -6px 0; border-radius: 7px; }}
QFrame#playerCard {{ background-color: {t.surface}; border: 1px solid {t.border}; border-radius: 8px; }}
QFrame#playerCard[active="true"] {{ border-color: {t.accent}; }}
QLabel#playerName {{ font-weight: 600; font-size: 14px; }}
QLabel#playerSub {{ color: {t.text_dim}; font-size: 11px; }}
QLabel#clock {{ background-color: {t.clock}; border: 1px solid {t.border}; border-radius: 6px; padding: 2px 10px;
    font-family: "Consolas", "Cascadia Mono", monospace; font-size: 24px; font-weight: 600; }}
QLabel#clock[state="active"] {{ background-color: {t.clock_active}; }}
QLabel#clock[state="low"] {{ background-color: {t.clock_low}; color: {t.danger}; }}
QFrame#engineOutput {{ background-color: {t.surface}; border: 1px solid {t.border}; border-radius: 8px; }}
QLabel#outputTitle {{ font-weight: 600; }}
QLabel#fieldName {{ color: {t.text_dim}; font-size: 11px; }}
QLabel#fieldValue {{ font-family: "Consolas", "Cascadia Mono", monospace; }}
"""


def palette(t: Theme) -> QPalette:
    """A palette matching the style sheet, for parts Qt draws itself."""
    p = QPalette()
    roles = {
        QPalette.ColorRole.Window: t.window, QPalette.ColorRole.WindowText: t.text,
        QPalette.ColorRole.Base: t.surface, QPalette.ColorRole.AlternateBase: t.surface_alt,
        QPalette.ColorRole.ToolTipBase: t.raised, QPalette.ColorRole.ToolTipText: t.text,
        QPalette.ColorRole.Text: t.text, QPalette.ColorRole.Button: t.raised,
        QPalette.ColorRole.ButtonText: t.text, QPalette.ColorRole.Highlight: t.accent,
        QPalette.ColorRole.HighlightedText: t.accent_text, QPalette.ColorRole.PlaceholderText: t.text_dim,
        QPalette.ColorRole.Link: t.accent,
    }
    for role, color in roles.items():
        p.setColor(role, QColor(color))
    for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text, QPalette.ColorRole.ButtonText):
        p.setColor(QPalette.ColorGroup.Disabled, role, QColor(t.text_dim))
    return p


def apply_theme(app: QApplication, name: str) -> Theme:
    """Makes `name` the application's theme (the default if unknown)."""
    global _current
    _current = THEMES.get(name, THEMES[DEFAULT_THEME])
    app.setStyle("Fusion")
    app.setPalette(palette(_current))
    app.setStyleSheet(style_sheet(_current))
    return _current


@dataclass(frozen=True)
class BoardTheme:
    """The board's square colors; highlights are shared by every theme."""

    name: str
    title: str
    light: str
    dark: str


BOARD_THEMES: dict[str, BoardTheme] = {t.name: t for t in (
    BoardTheme("brown", "Brown", "#f0d9b5", "#b58863"),
    BoardTheme("green", "Green", "#eeeed2", "#769656"),
    BoardTheme("blue", "Blue", "#dee3e6", "#8ca2ad"),
    BoardTheme("slate", "Slate", "#d9dde3", "#7d8796"),
    BoardTheme("purple", "Purple", "#ece6f2", "#9b7fb8"),
    BoardTheme("coral", "Coral", "#f4e3d7", "#d08a6f"),
)}
DEFAULT_BOARD_THEME = "brown"

# Highlights drawn over the squares (RGBA), the same on every board theme.
LAST_MOVE = (155, 199, 0, 105)
SELECTED = (20, 85, 30, 128)
TARGET = (20, 85, 30, 110)
PREMOVE = (20, 30, 85, 120)
HOVER = (255, 255, 255, 70)
ENGINE_ARROW = (21, 120, 27, 170)
THREAT_ARROW = (178, 34, 34, 170)  # what the opponent threatens (analysis after a pass)
# Shapes the user draws (right mouse button), by color name: plain, Shift,
# Ctrl, Alt.
SHAPE_COLORS = {"green": (21, 120, 27, 170), "red": (178, 34, 34, 170), "blue": (0, 64, 168, 170),
                "yellow": (230, 143, 0, 170)}
