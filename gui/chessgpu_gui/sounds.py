"""Sound effects, synthesized: nothing to ship or download.

Each effect is a few damped tones and noise bursts, written once per volume
step as a 16-bit mono WAV file in a cache folder, then played asynchronously
by Windows' built-in player (winsound; a new sound replaces one still
playing). Elsewhere the GUI stays silent. move_sound() picks the effect for
a move: check, castling, promotion, capture or a plain move.
"""

from __future__ import annotations

import io
import math
import pathlib
import random
import struct
import sys
import wave

from .rules import Move, Position

RATE = 22050
SOUNDS = ("move", "capture", "check", "castle", "promote", "start", "end", "low_time", "illegal")
VOLUME_STEPS = 10


def _tone(freq: float, seconds: float, decay: float, harmonics=((1.0, 1.0),)) -> list[float]:
    """A damped tone: partials (frequency ratio, amplitude) fading as exp(-decay * t)."""
    n = int(seconds * RATE)
    out = []
    for i in range(n):
        t = i / RATE
        attack = min(1.0, i / (0.002 * RATE))  # a 2 ms ramp, no click at the start
        v = sum(a * math.sin(2 * math.pi * freq * r * t) for r, a in harmonics)
        out.append(attack * math.exp(-decay * t) * v)
    return out


def _noise(seconds: float, decay: float, seed: int) -> list[float]:
    """A noise burst (fixed seed, so every build of a sound is identical)."""
    rng = random.Random(seed)
    n = int(seconds * RATE)
    ramp = 0.002 * RATE  # the same 2 ms attack as the tones
    return [rng.uniform(-1.0, 1.0) * min(1.0, i / ramp) * math.exp(-decay * i / RATE) for i in range(n)]


def _mix(*parts: tuple[float, float, list[float]]) -> list[float]:
    """Sums (start in seconds, gain, samples) tracks."""
    length = max(int(start * RATE) + len(s) for start, _gain, s in parts)
    out = [0.0] * length
    for start, gain, s in parts:
        offset = int(start * RATE)
        for i, v in enumerate(s):
            out[offset + i] += gain * v
    return out


def _thock(freq: float, seed: int) -> list[float]:
    """A piece set down on wood: a low partial-rich knock plus a short click."""
    body = _tone(freq, 0.10, 34.0, ((1.0, 1.0), (2.7, 0.35), (5.2, 0.15)))
    return _mix((0.0, 1.0, body), (0.0, 0.45, _noise(0.02, 220.0, seed)))


def samples(name: str) -> list[float]:
    """The effect's samples at full scale (peak 0.9)."""
    if name == "move":
        s = _thock(200.0, 1)
    elif name == "capture":
        hit = _tone(320.0, 0.13, 30.0, ((1.0, 1.0), (2.2, 0.5), (3.9, 0.3)))
        s = _mix((0.0, 1.0, hit), (0.0, 0.8, _noise(0.035, 120.0, 2)), (0.016, 0.6, _thock(260.0, 3)))
    elif name == "check":
        s = _mix((0.0, 1.0, _tone(740.0, 0.12, 18.0)), (0.11, 1.0, _tone(988.0, 0.16, 14.0)))
    elif name == "castle":
        s = _mix((0.0, 1.0, _thock(200.0, 4)), (0.09, 0.9, _thock(170.0, 5)))
    elif name == "promote":
        s = _mix(*((i * 0.07, 1.0, _tone(f, 0.14, 16.0)) for i, f in enumerate((523.25, 659.25, 783.99))))
    elif name == "start":
        s = _mix((0.0, 1.0, _tone(523.25, 0.38, 8.0)), (0.0, 0.8, _tone(783.99, 0.38, 8.0)))
    elif name == "end":
        s = _mix(*((i * 0.15, 1.0, _tone(f, 0.32, 8.0)) for i, f in enumerate((783.99, 659.25, 523.25))))
    elif name == "low_time":
        s = _tone(1200.0, 0.045, 80.0)
    elif name == "illegal":
        s = _tone(140.0, 0.16, 10.0, ((1.0, 1.0), (3.0, 0.33), (5.0, 0.2)))
    else:
        raise ValueError(f"unknown sound {name!r}")
    peak = max(abs(v) for v in s) or 1.0
    return [0.9 * v / peak for v in s]


def wav_bytes(name: str, volume: float = 1.0) -> bytes:
    """The effect as a WAV file (volume 0..1)."""
    volume = max(0.0, min(1.0, volume))
    data = b"".join(struct.pack("<h", int(round(32767 * volume * v))) for v in samples(name))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(data)
    return buf.getvalue()


def move_sound(before: Position, move: Move, after: Position) -> str:
    """The effect for `move`, played from `before` to `after`."""
    if after.in_check():
        return "check"
    if before.is_castling(move):
        return "castle"
    if move.promotion:
        return "promote"
    if before.is_capture(move):
        return "capture"
    return "move"


def _winsound_play(path: pathlib.Path) -> None:
    import winsound

    winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT)


class SoundPlayer:
    """Plays effects by name; the WAV files are made on first use, one per
    volume step. `backend` plays a file (replaceable: tests record calls)."""

    def __init__(self, folder: pathlib.Path, backend=None, enabled: bool = True, volume: float = 0.7):
        self.folder = pathlib.Path(folder)
        self.backend = backend if backend is not None else (_winsound_play if sys.platform == "win32" else None)
        self.enabled = enabled
        self.volume = volume

    def file(self, name: str) -> pathlib.Path:
        step = round(max(0.0, min(1.0, self.volume)) * VOLUME_STEPS)
        path = self.folder / f"{name}-{step}.wav"
        if not path.exists():
            self.folder.mkdir(parents=True, exist_ok=True)
            path.write_bytes(wav_bytes(name, step / VOLUME_STEPS))
        return path

    def play(self, name: str) -> None:
        if not self.enabled or self.backend is None or self.volume <= 0:
            return
        try:
            self.backend(self.file(name))
        except (OSError, RuntimeError):
            pass  # no sound device, or a file we couldn't write: stay silent
