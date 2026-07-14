"""Load / auto-build the native libboardgen.so."""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from pathlib import Path

_DIR = Path(__file__).resolve().parent
_LIB_PATH = _DIR / "libboardgen.so"

_FEN_MAX = 128


def _build() -> None:
    makefile = _DIR / "Makefile"
    if not makefile.exists():
        raise RuntimeError(f"Missing Makefile at {makefile}")
    subprocess.check_call(
        ["make", "-C", str(_DIR), "all"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    if not _LIB_PATH.exists():
        raise RuntimeError(f"Build succeeded but {_LIB_PATH} missing")


def _load() -> ctypes.CDLL:
    if not _LIB_PATH.exists():
        _build()
    lib = ctypes.CDLL(str(_LIB_PATH))

    lib.rbg_generate_fens_ex.argtypes = [
        ctypes.c_char_p,  # out
        ctypes.c_int,     # n
        ctypes.c_int,     # min_ply
        ctypes.c_int,     # max_ply
        ctypes.c_uint64,  # seed
        ctypes.c_int,     # skip_terminal
        ctypes.c_int,     # max_retries
    ]
    lib.rbg_generate_fens_ex.restype = ctypes.c_int

    lib.rbg_perft.argtypes = [ctypes.c_int]
    lib.rbg_perft.restype = ctypes.c_uint64

    lib.rbg_version.argtypes = []
    lib.rbg_version.restype = ctypes.c_char_p

    return lib


_LIB: ctypes.CDLL | None = None


def get_lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is None:
        _LIB = _load()
    return _LIB


def fen_max() -> int:
    return _FEN_MAX


def perft(depth: int) -> int:
    return int(get_lib().rbg_perft(int(depth)))


def version() -> str:
    return get_lib().rbg_version().decode("ascii")
