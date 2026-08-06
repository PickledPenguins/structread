"""One writer per output format, all with the same interface.

    writer = make_writer(fmt, path, schemas, indent)
    writer.write(record, context)   # once per result
    count = writer.close(stdout)    # flushes and reports what was written

``record`` maps output names to values; ``context`` carries the reserved fields
for that result, which is what the templated ``bin`` and ``npy`` paths use.

JSON and CSV buffer, because both need to be completed before they can be
written; the array formats stream to files immediately and buffer only when the
destination is standard output. Nothing but data ever reaches standard output,
so an array stream can be piped safely.
"""

from __future__ import annotations

import csv as csvmod
import io
import json
from typing import Any, cast

import numpy as np

from core import DataError, OutputError, UsageError, jsonable, log
from schema import Schema

# --------------------------------------------------------------------------
# Rendering values
# --------------------------------------------------------------------------

def describe_array(array: np.ndarray) -> dict:
    """Render an array as a JSON object: dtype, shape, and row-major values."""
    if array.dtype == np.dtype(object):
        dtype_name = "str"
        data = [jsonable(item) for item in array.reshape(-1).tolist()]
    else:
        dtype_name = str(array.dtype)
        data = [jsonable(item) for item in array.reshape(-1).tolist()]
    return {"dtype": dtype_name, "shape": list(array.shape), "data": data}


def render_value(value: Any) -> Any:
    """Render one output value as JSON, descending into lists and structures."""
    if isinstance(value, np.ndarray):
        return describe_array(value)
    if isinstance(value, dict):
        return {key: render_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [render_value(item) for item in value]
    return jsonable(value)


# --------------------------------------------------------------------------
# The writers
# --------------------------------------------------------------------------


class JsonWriter:
    """Collects every result and writes them as one JSON array."""

    name = "json"

    def __init__(self, path: str | None, indent: int = 2):
        self.path = path
        self.indent = indent
        self.records: list[dict] = []

    def write(self, record: dict[str, Any], context: dict) -> None:
        """Buffer one result."""
        self.records.append({key: render_value(value) for key, value in record.items()})

    def close(self, stdout) -> int:
        """Write the buffered results and report how many there were."""
        text = json.dumps(self.records, indent=self.indent or None) + "\n"
        _emit_text(self.path, text, stdout)
        return len(self.records)


class CsvWriter:
    """Writes one CSV row per result, with a single header line."""

    name = "csv"

    def __init__(self, path: str | None, columns: list[str]):
        self.path = path
        self.columns = columns
        self.buffer = io.StringIO()
        self.writer = csvmod.writer(self.buffer, lineterminator="\n")
        self.writer.writerow(columns)
        self.rows = 0

    def write(self, record: dict[str, Any], context: dict) -> None:
        """Write one row, rejecting any value that turned out to be an array."""
        row = []
        for column in self.columns:
            value = record.get(column)
            if isinstance(value, np.ndarray):
                raise DataError(f"{context['source']}: field {column!r} is an array; "
                                f"csv output holds single values only")
            row.append("" if value is None else jsonable(value))
        self.writer.writerow(row)
        self.rows += 1

    def close(self, stdout) -> int:
        """Write the accumulated table."""
        _emit_text(self.path, self.buffer.getvalue(), stdout)
        return self.rows


class BinWriter:
    """Writes array data, either to standard output or to templated files.

    In `bin` mode the array's bytes are written exactly as stored; in `npy` mode
    a numpy header carrying dtype, shape and byte order is written first, so the
    result reloads with numpy.load without needing the schema.
    """

    def __init__(self, path: str | None, name: str = "bin"):
        self.name = name
        self.path = path
        self.templated = bool(path) and ("{index}" in cast("str", path)
                                         or "{stem}" in cast("str", path))
        self.written: set[str] = set()
        self.count = 0
        self._stdout_chunks: list[bytes] = []

    def write(self, record: dict[str, Any], context: dict) -> None:
        """Write one array's bytes."""
        (name, value), = record.items()
        if not isinstance(value, np.ndarray):
            raise DataError(f"{context['source']}: field {name!r} is not an array; "
                            f"bin output writes array data only")
        payload = self._encode(value)
        if self.path is None:
            self._stdout_chunks.append(payload)
        else:
            target = (self.path.replace("{stem}", str(context["stem"]))
                      .replace("{index}", str(context["index"])))
            if target in self.written:
                raise UsageError(
                    f"more than one array would be written to {target}; add {{index}} or "
                    f"{{stem}} to the output path, or select one record with --index N")
            try:
                with open(target, "wb") as handle:
                    handle.write(payload)
            except OSError as exc:
                raise OutputError(f"cannot write {target}: {exc}") from exc
            self.written.add(target)
            log.debug("wrote %d byte(s) to %s", len(payload), target)
        self.count += 1

    def _encode(self, array: np.ndarray) -> bytes:
        """Turn an array into the bytes this writer emits."""
        if self.name == "bin":
            return np.ascontiguousarray(array).tobytes()
        buffer = io.BytesIO()
        np.save(buffer, np.ascontiguousarray(array), allow_pickle=False)
        return buffer.getvalue()

    def close(self, stdout) -> int:
        """Flush buffered bytes to standard output, if that is the destination."""
        if self.path is None and self._stdout_chunks:
            stream = getattr(stdout, "buffer", stdout)
            stream.write(b"".join(self._stdout_chunks))
            stream.flush()
        return self.count


def _emit_text(path: str | None, text: str, stdout) -> None:
    """Write text to a file, or to standard output when no path is given."""
    if path is None:
        stdout.write(text)
        return
    try:
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
    except OSError as exc:
        raise OutputError(f"cannot write {path}: {exc}") from exc


def make_writer(fmt: str, path: str | None, schemas: list[Schema], indent: int):
    """Construct the writer for the chosen output format."""
    if fmt in ("json", "csv") and path and ("{index}" in path or "{stem}" in path):
        raise UsageError(f"{fmt} output writes a single file; placeholders in {path!r} "
                         f"are only meaningful with --format bin or --format npy")
    if fmt == "json":
        return JsonWriter(path, indent)
    if fmt == "csv":
        return CsvWriter(path, [name for _, name in schemas[0].output.fields])
    return BinWriter(path, fmt)
