"""Engine profiles: an engine program plus the UCI options to start it with.
Two profiles of the same program with different options (e.g. Eval=pesto
and Eval=material) are two players, so variants can play each other."""

from __future__ import annotations

import json
import pathlib
from dataclasses import asdict, dataclass, field

ROOT = pathlib.Path(__file__).resolve().parents[2]
CHESSGPU = ROOT / "build" / "Release" / "chessgpu.exe"
STOCKFISH = ROOT / "tools" / "stockfish" / "stockfish" / "stockfish-windows-x86-64-universal.exe"


@dataclass
class EngineProfile:
    name: str
    path: str
    args: list[str] = field(default_factory=list)
    options: dict[str, str] = field(default_factory=dict)

    def key(self) -> tuple:
        """Profiles with the same key can share a running engine process."""
        return (str(pathlib.Path(self.path)), tuple(self.args), tuple(sorted(self.options.items())))


def default_profiles() -> list[EngineProfile]:
    profiles = [
        EngineProfile("ChessGPU", str(CHESSGPU), ["uci"]),
        EngineProfile("ChessGPU (material eval)", str(CHESSGPU), ["uci"], {"Eval": "material"}),
        EngineProfile("ChessGPU (warp model)", str(CHESSGPU), ["uci"], {"Model": "warp", "BlockSize": "64"}),
        EngineProfile("ChessGPU (CPU search)", str(CHESSGPU), ["uci"], {"Search": "cpu"}),
    ]
    if STOCKFISH.exists():
        profiles.append(EngineProfile("Stockfish", str(STOCKFISH)))
    return profiles


def profiles_to_json(profiles: list[EngineProfile]) -> str:
    return json.dumps([asdict(p) for p in profiles], indent=1)


def profiles_from_json(text: str) -> list[EngineProfile]:
    """Profiles from JSON; entries that don't parse are skipped."""
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return []
    out = []
    for d in data if isinstance(data, list) else []:
        try:
            out.append(EngineProfile(str(d["name"]), str(d["path"]), [str(a) for a in d.get("args", [])],
                                     {str(k): str(v) for k, v in d.get("options", {}).items()}))
        except (KeyError, TypeError, AttributeError):
            continue
    return out
