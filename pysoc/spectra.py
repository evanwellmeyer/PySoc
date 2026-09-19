"""Locate (and if needed download) the GA7 spectral files that Isca uses.

The files belong to SOCRATES (github.com/MetOffice/socrates, BSD-3-Clause) and are not
shipped with PySoc.  :func:`ga7_spectral_files` uses a local SOCRATES checkout when one
exists at ``reference/socrates`` (the development layout of this repository); otherwise
it downloads the files from GitHub, at the SOCRATES commit PySoc was validated against,
into a cache directory and checks their SHA-256 hashes.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path

SOCRATES_COMMIT = "76a675b2e7fc2d3d325ec3e6fed7a78f45e0cff6"
_URL = "https://raw.githubusercontent.com/MetOffice/socrates/{commit}/data/spectra/ga7/{name}"
GA7_SHA256 = {
    "sp_lw_ga7": "7e035282061e53ae7f43564bec79340ce333389d7d712b1c921a97abd8015fe8",
    "sp_lw_ga7_k": "e7460d10f2bfb4518e98c1cad3358295d4c4b7dd47f8b7601994523ec874c4a3",
    "sp_sw_ga7": "6de492dd302b60555acc278507bed6a1c5068c8d075fb9fc0320f8d30afc01d8",
    "sp_sw_ga7_k": "fa5b8ce8416cc433504d2b2f0c0b26904df8cfa137f76e4e8ce27fc61a53f4a4",
}
_LOCAL = Path(__file__).resolve().parents[1] / "reference" / "socrates" / "data" / "spectra" / "ga7"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def default_cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return Path(base) / "pysoc" / "ga7"


def ga7_spectral_files(directory: str | os.PathLike | None = None) -> tuple[str, str]:
    """Return the paths of ``sp_lw_ga7`` and ``sp_sw_ga7``, downloading them if needed.

    ``directory`` is where to look for / download the files; by default a local SOCRATES
    checkout is used if present, else ``~/.cache/pysoc/ga7``.
    """
    if directory is None:
        if all((_LOCAL / name).exists() for name in GA7_SHA256):
            return str(_LOCAL / "sp_lw_ga7"), str(_LOCAL / "sp_sw_ga7")
        directory = default_cache_dir()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, digest in GA7_SHA256.items():
        path = directory / name
        if path.exists() and _sha256(path) == digest:
            continue
        tmp = path.with_suffix(".part")
        with urllib.request.urlopen(_URL.format(commit=SOCRATES_COMMIT, name=name), timeout=60) as r:
            tmp.write_bytes(r.read())
        if _sha256(tmp) != digest:
            tmp.unlink()
            raise RuntimeError(f"downloaded {name} does not match the expected SHA-256")
        tmp.replace(path)
    return str(directory / "sp_lw_ga7"), str(directory / "sp_sw_ga7")
