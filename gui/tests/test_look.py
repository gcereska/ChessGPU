"""Look and feel: themes, sounds, the board's animation, shapes, premoves and
promotion overlay, player cards, the board area, and the view settings."""

import io
import struct
import wave

import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, QSettings, Qt, QTimer
from PySide6.QtGui import QColor, QMouseEvent
from PySide6.QtWidgets import QApplication, QCheckBox, QLabel, QRadioButton, QStyle, QStyleOptionButton

from chessgpu_gui.board_view import BoardView, plan_animation
from chessgpu_gui.dialogs import ANIMATION_SPEEDS, SettingsDialog, ViewSettings
from chessgpu_gui.game import GameController, TimeControl
from chessgpu_gui.move_tree import MoveTreeView
from chessgpu_gui.panels import BoardArea, EvalBar, NavigationBar, PlayerCard, figurine_san, material
from chessgpu_gui.pieces import VectorPieceSet
from chessgpu_gui.rules import BLACK, WHITE, Move, Position, parse_square
from chessgpu_gui.sounds import RATE, SOUNDS, SoundPlayer, move_sound, samples, wav_bytes
from chessgpu_gui.theme import (BOARD_THEMES, CHECK_MARK, DEFAULT_THEME, THEMES, apply_theme, current_theme,
                                palette, style_sheet, theme_colors)
from qt_helpers import Recorder, wait_until


def sq(name):
    return parse_square(name)


# --- Themes ---------------------------------------------------------------------------------------


def test_themes_share_their_tokens_and_build_style_sheets(qapp):
    names = {tuple(theme_colors(t)) for t in THEMES.values()}
    assert len(names) == 1  # every theme fills the same roles
    for t in THEMES.values():
        sheet = style_sheet(t)
        for token, color in theme_colors(t).items():
            assert QColor(color).isValid(), (t.name, token)
        # the colors a widget actually shows come from the theme
        for color in (t.window, t.surface, t.text, t.accent, t.border, t.clock_active, t.clock_low):
            assert color in sheet
        assert palette(t).color(palette(t).ColorRole.Window) == QColor(t.window)
    assert THEMES["dark"].window != THEMES["light"].window


def _indicator_colors(widget, element):
    """The colors inside a checkbox's or radio button's indicator."""
    widget.resize(widget.sizeHint())
    image = widget.grab().toImage()
    option = QStyleOptionButton()
    widget.initStyleOption(option)
    rect = widget.style().subElementRect(element, option, widget)
    return {image.pixelColor(x, y).name() for x in range(rect.left(), rect.right() + 1)
            for y in range(rect.top(), rect.bottom() + 1)}


def test_checkboxes_and_radio_buttons_show_in_both_themes(qapp):
    assert CHECK_MARK.is_file()
    try:
        for name in ("dark", "light"):
            t = apply_theme(qapp, name)
            for cls, element in ((QCheckBox, QStyle.SubElement.SE_CheckBoxIndicator),
                                 (QRadioButton, QStyle.SubElement.SE_RadioButtonIndicator)):
                off = cls("option")
                colors = _indicator_colors(off, element)
                assert colors - {QColor(t.window).name()}, (name, cls.__name__)  # a visible frame when unchecked
                on = cls("option")
                on.setChecked(True)
                assert QColor(t.accent).name() in _indicator_colors(on, element), (name, cls.__name__)
    finally:
        apply_theme(qapp, DEFAULT_THEME)


def test_apply_theme(qapp):
    try:
        t = apply_theme(qapp, "light")
        assert t is THEMES["light"] and current_theme() is t
        assert qapp.styleSheet() == style_sheet(t)
        # roles where both themes differ from the Fusion style's own palette
        for name in ("light", "dark"):
            t = apply_theme(qapp, name)
            roles = qapp.palette().ColorRole
            assert qapp.palette().color(roles.Window) == QColor(t.window)
            assert qapp.palette().color(roles.Highlight) == QColor(t.accent)
            assert qapp.palette().color(roles.Text) == QColor(t.text)
        assert apply_theme(qapp, "no such theme") is THEMES[DEFAULT_THEME]
    finally:
        apply_theme(qapp, DEFAULT_THEME)


def test_board_themes_are_valid(qapp):
    assert len(BOARD_THEMES) >= 6
    for t in BOARD_THEMES.values():
        assert QColor(t.light).isValid() and QColor(t.dark).isValid() and t.light != t.dark
        assert QColor(t.light).lightness() > QColor(t.dark).lightness()


# --- Sounds ---------------------------------------------------------------------------------------


def read_wav(data):
    with wave.open(io.BytesIO(data)) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, RATE)
        frames = w.readframes(w.getnframes())
    return [v for (v,) in struct.iter_unpack("<h", frames)]


@pytest.mark.parametrize("name", SOUNDS)
def test_every_sound_is_a_short_valid_wav(name):
    values = read_wav(wav_bytes(name, 1.0))
    seconds = len(values) / RATE
    assert 0.03 <= seconds <= 0.8, seconds
    peak = max(abs(v) for v in values)
    assert 0.85 * 32767 <= peak <= 0.91 * 32767  # normalized to 0.9
    assert abs(values[0]) < 0.1 * 32767  # no click at the start
    assert samples(name) == samples(name)  # deterministic (seeded noise)


def test_sound_volume():
    full = read_wav(wav_bytes("move", 1.0))
    half = read_wav(wav_bytes("move", 0.5))
    assert len(full) == len(half)
    assert max(map(abs, half)) == pytest.approx(max(map(abs, full)) / 2, rel=0.01)
    assert max(map(abs, read_wav(wav_bytes("move", 0.0)))) == 0
    assert max(map(abs, read_wav(wav_bytes("move", 7.0)))) == max(map(abs, full))  # clamped
    with pytest.raises(ValueError):
        samples("trumpet")


def test_move_sounds():
    start = Position.start()
    assert move_sound(start, Move.from_uci("e2e4"), start.push(Move.from_uci("e2e4"))) == "move"
    p = Position.from_fen("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1")
    assert move_sound(p, Move.from_uci("e4d5"), p.push(Move.from_uci("e4d5"))) == "capture"
    ep = Position.from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
    assert move_sound(ep, Move.from_uci("e5d6"), ep.push(Move.from_uci("e5d6"))) == "capture"
    castle = Position.from_fen("4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1")
    assert move_sound(castle, Move.from_uci("e1g1"), castle.push(Move.from_uci("e1g1"))) == "castle"
    promo = Position.from_fen("8/1P5k/8/8/8/8/8/4K3 w - - 0 1")
    assert move_sound(promo, Move.from_uci("b7b8n"), promo.push(Move.from_uci("b7b8n"))) == "promote"
    # a check outranks the rest (b8=Q gives check along the rank)
    promo_check = Position.from_fen("7k/1P6/8/8/8/8/8/4K3 w - - 0 1")
    assert move_sound(promo_check, Move.from_uci("b7b8q"), promo_check.push(Move.from_uci("b7b8q"))) == "check"


def test_sound_player(tmp_path):
    played = []
    player = SoundPlayer(tmp_path / "sounds", backend=played.append, volume=0.7)
    player.play("capture")
    assert [p.name for p in played] == ["capture-7.wav"] and played[0].exists()
    assert read_wav(played[0].read_bytes()) == read_wav(wav_bytes("capture", 0.7))
    player.volume = 0.31  # rounds to step 3: a file of its own
    player.play("capture")
    assert played[-1].name == "capture-3.wav"
    player.enabled = False
    player.play("move")
    player.enabled, player.volume = True, 0.0
    player.play("move")
    assert len(played) == 2  # disabled or muted: nothing

    def broken(path):
        raise OSError("no audio device")
    SoundPlayer(tmp_path / "s2", backend=broken).play("move")  # stays silent, no error
    SoundPlayer(tmp_path / "s3", backend=None).play("move")


# --- Board: animation -----------------------------------------------------------------------------


def plan(fen, uci, back=False):
    before = Position.from_fen(fen)
    after = before.push(Move.from_uci(uci))
    return plan_animation(after, before) if back else plan_animation(before, after)


def test_plan_animation():
    start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    p = plan(start, "g1f3")
    assert p.slides == [("N", sq("g1"), sq("f3"))] and not p.fade_out and not p.fade_in
    assert p.hidden() == {sq("f3")}
    capture = plan("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1", "e4d5")
    assert capture.slides == [("P", sq("e4"), sq("d5"))] and capture.fade_out == [("p", sq("d5"))]
    ep = plan("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6")
    assert ep.slides == [("P", sq("e5"), sq("d6"))] and ep.fade_out == [("p", sq("d5"))]
    castle = plan("4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1", "e1c1")
    assert sorted(castle.slides) == [("K", sq("e1"), sq("c1")), ("R", sq("a1"), sq("d1"))]
    promo = plan("1r5k/P7/8/8/8/8/8/4K3 w - - 0 1", "a7b8q")
    assert promo.slides == [("P", sq("a7"), sq("b8"))] and promo.fade_out == [("r", sq("b8"))]
    assert promo.hidden() == {sq("b8")}  # the queen shows once the pawn arrives
    # backwards: the piece slides back, a captured piece fades back in
    back = plan("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1", "e4d5", back=True)
    assert back.slides == [("P", sq("d5"), sq("e4"))] and back.fade_in == [("p", sq("d5"))]
    unpromote = plan("1r5k/P7/8/8/8/8/8/4K3 w - - 0 1", "a7b8q", back=True)
    assert unpromote.slides == [("Q", sq("b8"), sq("a7"))] and unpromote.fade_in == [("r", sq("b8"))]
    s = Position.from_fen(start)
    assert plan_animation(s, s) is None
    far = Position.from_fen("rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2")
    assert plan_animation(s, far) is None  # three moves apart: no animation


def test_board_animates_on_screen(qapp):
    board = BoardView(VectorPieceSet())
    board.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
    board.resize(320, 320)
    board.show()
    start = Position.start()
    board.set_position(start)
    board.animation_ms = 60
    board.set_position(start.push(Move.from_uci("e2e4")), Move.from_uci("e2e4"))
    assert board.animating
    board.grab()  # paints mid-slide
    assert wait_until(lambda: not board.animating, 2000)
    board.animation_ms = 0
    board.set_position(start)
    assert not board.animating
    board.animation_ms = 60
    board.set_position(start.push(Move.from_uci("d2d4")), animate=False)
    assert not board.animating
    board.set_position(start)
    assert board.animating
    board.hide()  # hiding stops it
    assert not board.animating


def test_off_screen_board_never_animates(qapp):
    board = BoardView(VectorPieceSet())
    board.set_position(Position.start().push(Move.from_uci("e2e4")))
    assert not board.animating


# --- Board: input ---------------------------------------------------------------------------------


def mouse(widget, kind, pos, button=Qt.MouseButton.LeftButton, buttons=None, modifiers=Qt.KeyboardModifier.NoModifier):
    buttons = button if buttons is None else buttons
    event = QMouseEvent(kind, QPointF(pos), QPointF(widget.mapToGlobal(QPoint(int(pos.x()), int(pos.y())))),
                        button, buttons, modifiers)
    QApplication.sendEvent(widget, event)


def center(board, name):
    return board.square_rect(sq(name)).center()


def click(board, name, button=Qt.MouseButton.LeftButton, modifiers=Qt.KeyboardModifier.NoModifier):
    p = center(board, name)
    mouse(board, QEvent.Type.MouseButtonPress, p, button, modifiers=modifiers)
    mouse(board, QEvent.Type.MouseButtonRelease, p, button, Qt.MouseButton.NoButton, modifiers=modifiers)


def right_drag(board, a, b, modifiers=Qt.KeyboardModifier.NoModifier):
    mouse(board, QEvent.Type.MouseButtonPress, center(board, a), Qt.MouseButton.RightButton, modifiers=modifiers)
    mouse(board, QEvent.Type.MouseButtonRelease, center(board, b), Qt.MouseButton.RightButton,
          Qt.MouseButton.NoButton, modifiers=modifiers)


@pytest.fixture
def board(qapp):
    b = BoardView(VectorPieceSet())
    b.resize(400, 400)
    b.moves = Recorder(b.move_made)
    return b


def test_shapes(board):
    changes = Recorder(board.shapes_changed)
    click(board, "e4", Qt.MouseButton.RightButton)
    assert board.shape_circles == {sq("e4"): "green"}
    click(board, "e4", Qt.MouseButton.RightButton, Qt.KeyboardModifier.ShiftModifier)
    assert board.shape_circles == {sq("e4"): "red"}  # another color replaces it
    click(board, "e4", Qt.MouseButton.RightButton, Qt.KeyboardModifier.ShiftModifier)
    assert board.shape_circles == {}  # the same color removes it
    right_drag(board, "g1", "f3")
    right_drag(board, "d1", "h5", Qt.KeyboardModifier.ControlModifier)
    right_drag(board, "a2", "a4", Qt.KeyboardModifier.AltModifier)
    assert board.shape_arrows == {(sq("g1"), sq("f3")): "green", (sq("d1"), sq("h5")): "blue",
                                  (sq("a2"), sq("a4")): "yellow"}
    right_drag(board, "g1", "f3")
    assert (sq("g1"), sq("f3")) not in board.shape_arrows
    board.grab()  # paints arrows and circles
    click(board, "e2")  # a left click clears every shape
    assert board.shape_arrows == {} and board.shape_circles == {}
    assert len(changes) == 8
    board.set_shapes({(sq("e2"), sq("e4")): "red"}, {sq("d5"): "blue"})
    assert board.shape_arrows and board.shape_circles
    assert board.moves.calls == []


def test_shapes_stay_out_of_edit_mode(board):
    board.edit_mode = True
    clicks = Recorder(board.square_clicked)
    click(board, "e4", Qt.MouseButton.RightButton)
    assert board.shape_circles == {} and len(clicks) == 1


def test_premoves(board):
    after_e4 = Position.start().push(Move.from_uci("e2e4"))  # Black to move: White may premove
    board.set_position(after_e4)
    board.set_interactive(False)
    changed = Recorder(board.premove_changed)
    click(board, "d2")
    click(board, "d4")
    assert board.premoves == [] and board.moves.calls == []  # premove_color not set: no premoves
    board.set_premove_color(WHITE)
    click(board, "g1")
    assert board.targets(sq("g1")) == {sq("f3"), sq("h3"), sq("e2")}
    click(board, "f3")
    assert board.premoves == [Move.from_uci("g1f3")] and len(changed) == 1 and board.moves.calls == []
    board.grab()  # paints the premove squares
    click(board, "a7")  # Black's piece: not ours to premove
    assert board._selected is None
    click(board, "c3", Qt.MouseButton.RightButton)  # cancels the premove
    assert board.premoves == [] and len(changed) == 2
    click(board, "d2")
    click(board, "d4")
    assert board.take_premove() == Move.from_uci("d2d4") and board.premoves == []
    assert board.take_premove() is None
    promo = Position.from_fen("4k3/1P6/8/8/8/8/7r/K7 b - - 0 1")
    board.set_position(promo)
    click(board, "b7")
    click(board, "b8")
    assert board.premoves == [Move.from_uci("b7b8q")]  # premoved promotions queen
    board.set_premove_color(None)
    assert board.premoves == [Move.from_uci("b7b8q")]  # kept until played or cancelled
    board.cancel_premoves()
    assert board.premoves == [] and board.moves.calls == []


def test_auto_queen_and_promotion_overlay(board):
    board.set_position(Position.from_fen("4k3/1P6/8/8/8/8/8/4K3 w - - 0 1"))
    board.set_interactive(True)
    board.auto_queen = True
    board.promotion_chooser = lambda color, s: pytest.fail("auto-queen must not ask")
    click(board, "b7")
    click(board, "b8")
    assert board.moves.calls == [(Move.from_uci("b7b8q"),)]
    board.auto_queen = False
    board.promotion_chooser = board._ask_promotion
    squares = board.promotion_squares(WHITE, sq("b8"))
    assert [letter for letter, _ in squares] == list("qnrb")
    assert squares[1][1].top() > squares[0][1].top()  # toward the middle from the top edge
    # the overlay waits in its own event loop: answer with a click on the knight
    knight = squares[1][1].center()
    QTimer.singleShot(50, lambda: mouse(board, QEvent.Type.MouseButtonPress, knight))
    assert board._ask_promotion(WHITE, sq("b8")) == "n"
    QTimer.singleShot(50, lambda: mouse(board, QEvent.Type.MouseButtonPress, center(board, "h1")))
    assert board._ask_promotion(WHITE, sq("b8")) is None  # a click outside cancels
    board.set_flipped(True)
    assert board.promotion_squares(WHITE, sq("b8"))[1][1].top() < board.promotion_squares(WHITE, sq("b8"))[0][1].top()


def test_hover_and_board_theme(board):
    board.set_interactive(True)
    mouse(board, QEvent.Type.MouseButtonPress, center(board, "g1"))
    mouse(board, QEvent.Type.MouseMove, center(board, "f3"), Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton)
    assert board._hover == sq("f3")
    mouse(board, QEvent.Type.MouseMove, center(board, "f4"), Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton)
    assert board._hover is None  # not a target
    mouse(board, QEvent.Type.MouseButtonRelease, center(board, "f4"), Qt.MouseButton.LeftButton,
          Qt.MouseButton.NoButton)
    assert board._hover is None
    board.set_board_theme(BOARD_THEMES["green"])
    board.show_coordinates = False
    image = board.grab().toImage()
    corner = board.square_rect(sq("a3")).topLeft() + QPointF(2, 2)  # a3: a dark square, no piece
    assert image.pixelColor(corner.toPoint()) == QColor(BOARD_THEMES["green"].dark)


# --- Panels ---------------------------------------------------------------------------------------


def test_figurine_san():
    assert figurine_san("Nf3") == "♘f3"
    assert figurine_san("Qxe7+") == "♕xe7+"
    assert figurine_san("e8=Q#") == "e8=♕#"
    assert figurine_san("exd8=N") == "exd8=♘"
    assert figurine_san("O-O") == "O-O" and figurine_san("e4") == "e4"


def test_material():
    ahead, lead = material(Position.start())
    assert ahead == {WHITE: [], BLACK: []} and lead == 0
    # White has an extra knight, Black an extra pawn: White leads by 2
    ahead, lead = material(Position.from_fen("4k3/pp6/8/8/8/8/P7/1N2K3 w - - 0 1"))
    assert ahead == {WHITE: ["N"], BLACK: ["P"]} and lead == 2
    # a promoted queen counts as a queen
    ahead, lead = material(Position.from_fen("Q3k3/8/8/8/8/8/8/4K3 b - - 0 1"))
    assert ahead[WHITE] == ["Q"] and lead == 9


def test_player_card_material(qapp):
    g = GameController()
    card = PlayerCard(g, VectorPieceSet(), BLACK)
    card.set_position(Position.from_fen("4k3/pp3r2/8/8/8/8/P7/1N2K3 w - - 0 1"))  # Black up R+P for N
    assert card.material.kinds == ["R", "P"] and card.material.text() == "+3"
    card.set_color(WHITE)
    card.set_position(Position.from_fen("4k3/pp3r2/8/8/8/8/P7/1N2K3 w - - 0 1"))
    assert card.material.kinds == ["N"] and card.material.text() == ""
    card.grab()


def test_board_area_geometry(qapp):
    board, bar = BoardView(VectorPieceSet()), EvalBar()
    top, bottom = QLabel("top"), QLabel("bottom")
    area = BoardArea(board, bar, top, bottom)
    area.resize(900, 700)
    area._arrange()
    assert board.width() == board.height() and board.width() % 8 == 0
    assert top.width() == bottom.width() == bar.width() + BoardArea.GAP + board.width()
    assert top.geometry().bottom() < bar.geometry().top() and bottom.geometry().top() > board.geometry().bottom()
    assert board.geometry().top() == bar.geometry().top()
    area.set_cards(bottom, top)  # flipped: the cards trade places
    assert bottom.geometry().top() < top.geometry().top()
    area.resizeEvent(None)
    assert area.maximumWidth() == 2 * BoardArea.MARGIN + bar.width() + BoardArea.GAP + board.width()


def test_navigation_bar(qapp):
    nav = NavigationBar()
    seen = []
    for name in ("first", "previous", "next", "last"):
        getattr(nav, name).connect(lambda n=name: seen.append(n))
        nav.buttons[name].click()
    assert seen == ["first", "previous", "next", "last"]
    assert all(not b.icon().isNull() for b in nav.buttons.values())


def test_move_list_figurines(qapp):
    g = GameController()
    record = g.record
    for uci in ("g1f3", "g8f6"):
        record.push(Move.from_uci(uci))
    view = MoveTreeView()
    view.figurines = True
    view.set_tree(record, record.end)
    assert [text for _, text in view.anchors()] == ["♘f3", "♘f6"]


# --- Settings -------------------------------------------------------------------------------------


def test_view_settings_round_trip(qapp, tmp_path):
    s = QSettings(str(tmp_path / "gui.ini"), QSettings.Format.IniFormat)
    assert ViewSettings.load(s) == ViewSettings()
    v = ViewSettings(theme="light", board_theme="green", piece_set="vector", animation_ms=90, sounds=False,
                     volume=35, show_legal=False, coordinates=False, highlight_last=False, premoves=False,
                     auto_queen=True, figurines=True)
    v.save(s)
    s.sync()
    again = QSettings(str(tmp_path / "gui.ini"), QSettings.Format.IniFormat)
    assert ViewSettings.load(again) == v
    for key, value in (("theme", "neon"), ("board_theme", "plaid"), ("piece_set", "emoji"), ("volume", "250"),
                       ("animation_ms", "lots"), ("sounds", "maybe")):
        again.setValue(key, value)
    loaded = ViewSettings.load(again)
    defaults = ViewSettings()
    assert (loaded.theme, loaded.board_theme, loaded.piece_set) == (defaults.theme, defaults.board_theme,
                                                                    defaults.piece_set)
    # out of range: clamped; unreadable: the default
    assert loaded.volume == 100 and loaded.animation_ms == defaults.animation_ms and loaded.sounds is False


def test_settings_dialog(qapp):
    v = ViewSettings(theme="light", board_theme="blue", piece_set="glyphs", animation_ms=ANIMATION_SPEEDS["Slow"],
                     sounds=True, volume=40, show_legal=True, coordinates=False, highlight_last=True, premoves=False,
                     auto_queen=True, figurines=True)
    dialog = SettingsDialog(v)
    assert dialog.result_settings() == v
    dialog.sounds.setChecked(False)
    assert not dialog.volume.isEnabled()
    dialog.theme.setCurrentIndex(dialog.theme.findData("dark"))
    assert dialog.result_settings().theme == "dark" and dialog.result_settings().sounds is False
    custom = SettingsDialog(ViewSettings(animation_ms=123))
    assert custom.result_settings().animation_ms == 123  # a speed set in the file survives
