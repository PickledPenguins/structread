"""Vocabulary shared by every other module: errors, logging and types.

This module is the bottom of the dependency graph and imports nothing from the
rest of the tool, so anything here is safe to use anywhere. It holds:

* the exception classes, one per kind of failure, each mapped to an exit code
  by ``structread.main``;
* logging setup, including the ``TRACE`` level used by ``-vv``;
* the type table and the conversions that turn raw bytes or text into Python
  values;
* the constant tables that name what the tool accepts - formats, reserved
  fields, and the schema keys the validator recognises.

Keeping the key tables here rather than inline in the validator means the
accepted vocabulary can be read in one place instead of being reconstructed
from a dozen call sites.
"""

from __future__ import annotations

import logging
import math
import re
import sys
from typing import Any, cast

import numpy as np

__version__ = "0.5"

# Exit codes. Every failure path in the tool ends at one of these; see the
# exception classes below for which condition maps to which.
EXIT_OK = 0
EXIT_DATA = 1
EXIT_SCHEMA = 2
EXIT_USAGE = 3
EXIT_DEPENDENCY = 4

# Level below DEBUG used for per-field reads, enabled by -vv. Standard logging
# has no level between NOTSET and DEBUG, so one is registered here.
TRACE = 5

log = logging.getLogger("structread")


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------

class StructReadError(Exception):
    """Base class for every error raised by this tool."""


class SchemaError(StructReadError):
    """A schema is malformed, or refers to something that does not exist."""


class DataError(StructReadError):
    """An input file does not match its schema, or cannot be read."""


class OutputError(StructReadError):
    """An output file could not be written."""


class UsageError(StructReadError):
    """The combination of command-line arguments is not usable."""


class DependencyError(StructReadError):
    """A library required by an input format is not installed."""


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------

class _Formatter(logging.Formatter):
    """Compact console format: level name padded, then the message."""

    def format(self, record: logging.LogRecord) -> str:
        """Render one log record."""
        name = "TRACE" if record.levelno == TRACE else record.levelname
        return f"{name:<7} {record.getMessage()}"


def setup_logging(quiet: bool, verbosity: int, log_file: str | None) -> None:
    """Configure the console handler, and optionally a debug-level file handler."""
    logging.addLevelName(TRACE, "TRACE")
    log.handlers.clear()
    if quiet:
        console_level = logging.WARNING
    elif verbosity >= 2:
        console_level = TRACE
    elif verbosity == 1:
        console_level = logging.DEBUG
    else:
        console_level = logging.INFO
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(console_level)
    console.setFormatter(_Formatter())
    log.addHandler(console)
    lowest = console_level
    if log_file:
        try:
            handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        except OSError as exc:
            raise UsageError(f"cannot open log file {log_file}: {exc}") from exc
        handler.setLevel(TRACE)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
        log.addHandler(handler)
        lowest = min(lowest, TRACE)
    log.setLevel(lowest)


def trace(message: str) -> None:
    """Log at the per-field tracing level."""
    log.log(TRACE, message)


# --------------------------------------------------------------------------
# Type system
# --------------------------------------------------------------------------

# name -> (struct format character, numpy dtype name, size in bytes)
NUMERIC_TYPES: dict[str, tuple[str, str, int]] = {
    "i8": ("b", "int8", 1),
    "u8": ("B", "uint8", 1),
    "i16": ("h", "int16", 2),
    "u16": ("H", "uint16", 2),
    "i32": ("i", "int32", 4),
    "u32": ("I", "uint32", 4),
    "i64": ("q", "int64", 8),
    "u64": ("Q", "uint64", 8),
    "f32": ("f", "float32", 4),
    "f64": ("d", "float64", 8),
    "bool": ("?", "bool", 1),
}

BIN_TYPES = set(NUMERIC_TYPES) | {"char"}
TEXT_TYPES = set(NUMERIC_TYPES) | {"str"}
ENDIAN_CODES = {"little": "<", "big": ">", "native": "="}

RESERVED_FIELDS = ("source", "stem", "index", "schema")
INPUT_FORMATS = ("bin", "hdf5", "csv")
OUTPUT_FORMATS = ("json", "csv", "bin", "npy")

# Every key the schema loader accepts, in one place: the validator checks
# against these and the documentation tests enumerate them, so a key can never
# be added to the code without appearing in the README.
SCHEMA_KEYS_REQUIRED = ("name", "input")
SCHEMA_KEYS_OPTIONAL = ("version", "source", "description", "aliases", "output")
BIN_INPUT_KEYS = ("endian", "header", "record", "record_count", "record_size")
HDF5_INPUT_KEYS: tuple[str, ...] = ()
CSV_INPUT_KEYS = ("header", "delimiter", "skip_lines")
BIN_FIELD_KEYS = ("name", "type", "shape", "expect", "enum", "slice")
BIN_STRUCT_KEYS = ("name", "fields", "count")
BIN_DIRECTIVES = ("pad", "seek")
HDF5_FIELD_KEYS = ("at", "name", "type", "shape", "expect", "enum", "slice")
CSV_FIELD_KEYS = ("column", "columns", "name", "type", "row", "shape",
                  "expect", "enum", "slice")
OUTPUT_KEYS = ("format", "path", "fields")
OUTPUT_FIELD_KEYS = ("field", "as")
# Formats that carry array data and therefore take exactly one array field.
ARRAY_OUTPUTS = ("bin", "npy")
NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def numpy_dtype(type_name: str, endian: str = "=") -> np.dtype:
    """Return the numpy dtype for a schema type name and endianness code."""
    if type_name == "char":
        return np.dtype("uint8")
    dtype = np.dtype(NUMERIC_TYPES[type_name][1])
    return dtype.newbyteorder(cast("Any", endian)) if dtype.itemsize > 1 else dtype


def type_size(type_name: str) -> int:
    """Return the byte width of one element of a binary type."""
    return 1 if type_name == "char" else NUMERIC_TYPES[type_name][2]


def convert_text(text: str, type_name: str, where: str) -> Any:
    """Convert one piece of text to the requested type."""
    if type_name == "str":
        return text
    stripped = text.strip()
    if stripped == "":
        return None
    try:
        if type_name == "bool":
            lowered = stripped.lower()
            if lowered in ("1", "true", "yes", "y", "t"):
                return True
            if lowered in ("0", "false", "no", "n", "f"):
                return False
            raise ValueError("not a boolean")
        if type_name.startswith("f"):
            return float(stripped)
        return int(stripped, 0)
    except ValueError as exc:
        raise DataError(f"{where}: cannot read {text!r} as {type_name} ({exc})") from exc


# --------------------------------------------------------------------------
# Value conversion
# --------------------------------------------------------------------------

def jsonable(value: Any) -> Any:
    """Convert numpy scalars and byte strings into plain Python values.

    Everything that leaves a reader passes through here, so the writers only
    ever see types the standard library can serialise.
    """
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return [jsonable(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        # JSON has no literal for NaN or infinity; a missing value is null.
        return None
    return value
