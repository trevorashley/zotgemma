"""Find the Zotero data directory the way Zotero itself does.

Order: explicit path (``--zotero-dir``), ``ZOTGEMMA_ZOTERO_DIR``, the ``extensions.zotero.dataDir``
preference of the default profile (``profiles.ini`` then ``prefs.js``), then ``~/Zotero``.
Everything here is pure or read-only; nothing under the Zotero directory is written.
"""

from __future__ import annotations

import configparser
import json
import os
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path


class DiscoveryError(RuntimeError):
    """Raised when no usable Zotero data directory can be found."""


@dataclass(slots=True)
class Discovered:
    """Result of data-directory discovery."""

    path: Path
    source: str  # human-readable: "--zotero-dir", "ZOTGEMMA_ZOTERO_DIR", "extensions.zotero.dataDir in ...", ...
    prefs: dict[str, str] = field(default_factory=dict)  # extensions.zotero.* prefs of the default profile


def profiles_ini_path(home: Path | None = None, platform: str | None = None) -> Path:
    """Location of Zotero's ``profiles.ini`` for this platform."""
    home = home or Path.home()
    platform = platform or sys.platform
    if platform == "darwin":
        return home / "Library" / "Application Support" / "Zotero" / "profiles.ini"
    return home / ".zotero" / "zotero" / "profiles.ini"


def default_profile_dir(ini: Path) -> Path | None:
    """Profile directory marked ``Default=1`` (or the only profile) in ``profiles.ini``; None if absent."""
    if not ini.is_file():
        return None
    cp = configparser.ConfigParser(interpolation=None, strict=False)
    cp.optionxform = str  # keep key case
    try:
        cp.read(ini, encoding="utf-8")
    except (configparser.Error, OSError):
        return None
    profiles = [s for s in cp.sections() if s.lower().startswith("profile") and cp.has_option(s, "Path")]
    chosen = next((s for s in profiles if cp.get(s, "Default", fallback="0").strip() == "1"), None)
    if chosen is None and len(profiles) == 1:
        chosen = profiles[0]
    if chosen is None:
        return None
    raw = cp.get(chosen, "Path").strip()
    relative = cp.get(chosen, "IsRelative", fallback="1").strip() != "0"
    return (ini.parent / raw) if relative else Path(raw)


_PREF_RE = re.compile(
    r'user_pref\(\s*"(extensions\.zotero\.[^"]+)"\s*,\s*("(?:[^"\\]|\\.)*"|true|false|-?\d+)\s*\)\s*;')


def parse_prefs(text: str) -> dict[str, str]:
    """Extract ``extensions.zotero.*`` prefs from ``prefs.js`` text. Values are strings (``true``/``false`` for booleans)."""
    out: dict[str, str] = {}
    for name, raw in _PREF_RE.findall(text):
        if raw.startswith('"'):
            try:
                out[name] = json.loads(raw)
            except ValueError:
                out[name] = raw[1:-1]
        else:
            out[name] = raw
    return out


def read_profile_prefs(home: Path | None = None, platform: str | None = None) -> tuple[dict[str, str], Path | None]:
    """Prefs of the default profile and the ``prefs.js`` path that was read (or None)."""
    prof = default_profile_dir(profiles_ini_path(home, platform))
    if prof is None:
        return {}, None
    prefs_js = prof / "prefs.js"
    try:
        return parse_prefs(prefs_js.read_text(encoding="utf-8", errors="replace")), prefs_js
    except OSError:
        return {}, None


def discover_zotero_dir(flag: str | Path | None = None, env: Mapping[str, str] | None = None,
                        home: Path | None = None, platform: str | None = None) -> Discovered:
    """Resolve the Zotero data directory.

    An explicit source (flag or environment variable) is final: if it does not contain
    ``zotero.sqlite`` the error names it instead of silently trying another place.
    Raises :class:`DiscoveryError` listing every place looked when nothing works.
    """
    env = os.environ if env is None else env
    home = home or Path.home()
    prefs, prefs_js = read_profile_prefs(home, platform)

    def ok(p: Path) -> bool:
        return (p / "zotero.sqlite").is_file()

    for label, val in (("--zotero-dir", flag), ("ZOTGEMMA_ZOTERO_DIR", env.get("ZOTGEMMA_ZOTERO_DIR"))):
        if val:
            p = Path(val).expanduser()
            if ok(p):
                return Discovered(p, label, prefs)
            raise DiscoveryError(f"{label} is {p}, but it contains no zotero.sqlite. "
                                 f"Point it at the Zotero data directory (the folder holding zotero.sqlite and storage/).")

    looked: list[str] = []
    pref_dir = prefs.get("extensions.zotero.dataDir")
    # Zotero's default for useDataDir is false (and prefs.js omits default-valued prefs), so a leftover
    # dataDir line without useDataDir=true is stale and must be ignored.
    if pref_dir and prefs.get("extensions.zotero.useDataDir") == "true":
        p = Path(pref_dir).expanduser()
        if ok(p):
            return Discovered(p, f"extensions.zotero.dataDir in {prefs_js}", prefs)
        looked.append(f"extensions.zotero.dataDir = {p} (from {prefs_js}): no zotero.sqlite there")
    elif prefs_js:
        looked.append(f"{prefs_js}: no custom data directory in use (extensions.zotero.useDataDir is not true)")
    else:
        looked.append(f"{profiles_ini_path(home, platform)}: not found, or no default profile with a prefs.js")
    default = home / "Zotero"
    if ok(default):
        return Discovered(default, "default ~/Zotero", prefs)
    looked.append(f"{default}: no zotero.sqlite there")
    raise DiscoveryError("Could not find a Zotero data directory. Looked at:\n  - --zotero-dir / "
                         "ZOTGEMMA_ZOTERO_DIR: not set\n  - " + "\n  - ".join(looked)
                         + "\nSet ZOTGEMMA_ZOTERO_DIR or pass --zotero-dir.")
