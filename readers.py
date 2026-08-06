"""One reader per input format, all with the same interface.

    reader = Reader(schema, path, needed)
    for index, values in reader.iter_results(wanted):
        ...

``values`` maps full field paths to Python scalars, strings, numpy arrays or
lists. ``wanted`` is the index to read, or ``None`` for every index. ``needed``
is the set of field paths the output actually requires, including anything an
expression depends on; readers use it to skip work - the binary reader advances
past an unneeded array without materialising it, and the HDF5 reader never
opens a dataset nobody asked for.

Fields are always visited in declaration order, whatever the output needs, so
an expression can rely on the values it references having been read already.
"""

from __future__ import annotations

import csv as csvmod
import mmap
import os
import re
import struct
from collections.abc import Iterator
from typing import Any

import numpy as np

from core import (
    NUMERIC_TYPES,
    DataError,
    DependencyError,
    SchemaError,
    convert_text,
    jsonable,
    log,
    numpy_dtype,
    trace,
    type_size,
)
from expressions import apply_slice
from schema import Field, Schema

# --------------------------------------------------------------------------
# Binary
# --------------------------------------------------------------------------

class BinReader:
    """Reads fixed and repeating record structures out of a binary file."""

    def __init__(self, schema: Schema, path: str, needed: set[str]):
        self.schema = schema
        self.path = path
        self.needed = needed
        self.layout: list[tuple[str, int, int]] = []   # path, offset, byte size

    def iter_results(self, wanted: int | None) -> Iterator[tuple[int, dict[str, Any]]]:
        """Yield one (index, values) pair per record, or one pair with no records."""
        try:
            size = os.path.getsize(self.path)
        except OSError as exc:
            raise DataError(f"{self.path}: cannot stat file: {exc}") from exc
        if size == 0:
            raise DataError(f"{self.path}: file is empty")
        try:
            handle = open(self.path, "rb")  # noqa: SIM115 (closed in the finally below)
        except OSError as exc:
            raise DataError(f"{self.path}: cannot open file: {exc}") from exc
        try:
            buffer = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        except (OSError, ValueError) as exc:
            handle.close()
            raise DataError(f"{self.path}: cannot map file: {exc}") from exc
        try:
            yield from self._walk(buffer, size, wanted)
        finally:
            try:
                buffer.close()
            except BufferError:
                # A zero-copy view is still referenced, typically by an in-flight
                # traceback; releasing the mapping is left to the collector so the
                # original error is not masked.
                pass
            handle.close()

    def _walk(self, buffer, size: int, wanted: int | None) -> Iterator[tuple[int, dict]]:
        """Read the header once, then walk records."""
        base: dict[str, Any] = {}
        offset = 0
        header = self.schema.index.get("header")
        if header is not None:
            offset = self._read_fields(buffer, offset, header.children, base)
            log.info("%s: %d bytes, header read (%d bytes consumed)", self.path, size, offset)
        record = self.schema.index.get("record")
        if record is None:
            yield 0, base
            return

        total: int | None = None
        if self.schema.record_count is not None:
            total = self.schema.record_count.as_int(base, "record_count")
            log.info("%s: record_count = %d", self.path, total)
        stride: int | None = None
        if self.schema.record_size is not None:
            stride = self.schema.record_size.as_int(base, "record_size")
            if stride == 0:
                raise SchemaError("record_size evaluated to 0; this schema would loop forever")
            log.debug("%s: fixed record_size = %d", self.path, stride)

        first = offset
        # A declared record_size makes every record the same length, so record N
        # can be reached directly instead of by walking the ones before it.
        if stride is not None and wanted is not None:
            available = (size - first) // stride
            if total is not None:
                available = min(available, total)
            if wanted >= available:
                raise DataError(f"{self.path}: index {wanted} is past the last record "
                                f"({available} record(s) present)")
            values = dict(base)
            log.debug("%s: seeking directly to record %d at offset %d",
                      self.path, wanted, first + wanted * stride)
            self._read_fields(buffer, first + wanted * stride, record.children, values)
            yield wanted, values
            return

        produced = 0
        index = 0
        while True:
            if total is not None and index >= total:
                break
            if total is None and offset >= size:
                break
            if wanted is not None and index > wanted:
                break
            values = dict(base)
            start = offset
            try:
                offset = self._read_fields(buffer, offset, record.children, values)
            except DataError as exc:
                if total is None and start == offset:
                    break
                raise DataError(f"{self.path}: record {index}: {exc}") from exc
            short_slot = False
            if stride is not None:
                used = offset - start
                if used > stride:
                    raise DataError(f"{self.path}: record {index} consumed {used} byte(s), "
                                    f"more than the declared record_size of {stride}")
                offset = start + stride
                short_slot = offset > size and total is None
            log.debug("%s: record %d at offset %d, size %d", self.path, index, start,
                      offset - start)
            if offset == start:
                raise SchemaError("record consumed 0 bytes; this schema would loop forever")
            if wanted is None or index == wanted:
                produced += 1
                yield index, values
            index += 1
            if short_slot:
                log.warning("%s: the last record slot is short by %d byte(s)",
                            self.path, offset - size)
                break
        if wanted is not None and produced == 0:
            raise DataError(f"{self.path}: index {wanted} is past the last record "
                            f"({index} record(s) present)")
        # Only meaningful when the whole file was walked; stopping early at a
        # requested index leaves unread bytes by design.
        if total is None and wanted is None and offset < size:
            log.warning("%s: %d trailing byte(s) after the last record", self.path, size - offset)

    def _read_fields(self, buffer, offset: int, fields: list[Field],
                     values: dict[str, Any], element: dict | None = None) -> int:
        """Read one level of fields, recursing into nested structures.

        When `element` is given, values are also written into it under their
        short names, which is how one item of a repeating structure is built.
        """
        for field in fields:
            if field.kind == "pad":
                self._require(buffer, offset, field.spec["pad"], "padding")
                offset += field.spec["pad"]
            elif field.kind == "seek":
                assert field.count is not None
                target = field.count.as_int(values, "seek")
                if target > len(buffer):
                    raise DataError(f"seek to offset {target} is past the end of the file "
                                    f"({len(buffer)} byte(s))")
                trace(f"{self.path}: seek {offset} -> {target}")
                offset = target
            elif field.kind == "struct":
                nested: dict[str, Any] | None = {} if element is not None else None
                offset = self._read_fields(buffer, offset, field.children, values, nested)
                if element is not None:
                    element[field.name] = nested
            elif field.kind == "list":
                offset = self._read_list(buffer, offset, field, values, element)
            else:
                offset = self._read_value(buffer, offset, field, values, element)
        return offset

    def _read_list(self, buffer, offset: int, field: Field, values: dict[str, Any],
                   element: dict | None) -> int:
        """Read a repeating structure into a list of items."""
        assert field.count is not None
        count = field.count.as_int(values, f"structure {field.path} count")
        items: list[dict] = []
        for position in range(count):
            item: dict[str, Any] = {}
            trace(f"{self.path}: {field.path}[{position}] at offset {offset}")
            offset = self._read_fields(buffer, offset, field.children, values, item)
            items.append(item)
        values[field.path] = items
        if element is not None:
            element[field.name] = items
        self._collect_columns(field, items, values)
        return offset

    def _collect_columns(self, field: Field, items: list[dict], values: dict[str, Any]) -> None:
        """Expose each member of a repeating structure as one value per item."""
        for child in field.children:
            if child.kind in ("pad", "seek"):
                continue
            if child.kind == "struct":
                self._collect_columns(child, [item.get(child.name) or {} for item in items],
                                      values)
                values[child.path] = [item.get(child.name) for item in items]
            else:
                values[child.path] = [item.get(child.name) for item in items]

    def _read_value(self, buffer, offset: int, field: Field, values: dict[str, Any],
                    element: dict | None = None) -> int:
        """Read one scalar, string or array field."""
        assert field.type is not None
        element_size = type_size(field.type)
        if field.kind == "scalar":
            self._require(buffer, offset, element_size, f"field {field.path}")
            code = NUMERIC_TYPES[field.type][0]
            raw = struct.unpack_from(self.schema.endian + code, buffer, offset)[0]
            value = jsonable(raw)
            self._expect(field, value, f"offset {offset}")
            value = self._enum(field, value)
            self._store(field, value, values, element)
            self.layout.append((field.path, offset, element_size))
            trace(f"{self.path}: {field.path} = {value!r} at offset {offset}")
            return offset + element_size

        assert field.shape is not None
        dims = [item.as_int(values, f"field {field.path} shape") for item in field.shape]
        count = 1
        for dim in dims:
            count *= dim
        total = count * element_size
        self._require(buffer, offset, total, f"field {field.path}")
        start = offset

        if field.kind == "string":
            text = bytes(buffer[start:start + total]).split(b"\x00", 1)[0]
            value = text.decode("utf-8", errors="replace")
            self._expect(field, value, f"offset {offset}")
            self._store(field, value, values, element)
            self.layout.append((field.path, start, total))
            trace(f"{self.path}: {field.path} = {value!r} at offset {offset}")
        elif field.path in self.needed or element is not None:
            dtype = numpy_dtype(field.type, self.schema.endian)
            array = np.frombuffer(buffer, dtype=dtype, count=count, offset=start).copy()
            if len(dims) > 1:
                array = array.reshape(dims)
            array = apply_slice(array, field.slice, f"{self.path}: field {field.path}")
            self._store(field, array, values, element)
            self.layout.append((field.path, start, total))
            trace(f"{self.path}: {field.path} shape {list(array.shape)} at offset {start}")
        else:
            self.layout.append((field.path, start, total))
            trace(f"{self.path}: {field.path} skipped ({total} bytes, not in output)")
        return start + total

    @staticmethod
    def _store(field: Field, value: Any, values: dict[str, Any], element: dict | None) -> None:
        """Record a value under its full path, and in the current item if there is one."""
        values[field.path] = value
        if element is not None:
            element[field.name] = value

    def _expect(self, field: Field, value: Any, where: str) -> None:
        """Compare a value against the field's declared constant."""
        if field.expect is None or value == field.expect:
            return
        raise DataError(f"field {field.path} at {where}: expected {field.expect!r}, "
                        f"found {value!r}")

    def _enum(self, field: Field, value: Any) -> Any:
        """Map a raw value through the field's enum table, if it has one."""
        if field.enum is None:
            return value
        if value in field.enum:
            return field.enum[value]
        raise DataError(f"field {field.path}: value {value!r} is not in the enum table "
                        f"({', '.join(repr(key) for key in field.enum)})")

    @staticmethod
    def _require(buffer, offset: int, need: int, what: str) -> None:
        """Raise if fewer than `need` bytes remain."""
        if offset + need > len(buffer):
            raise DataError(f"{what} needs {need} byte(s) at offset {offset} but only "
                            f"{max(0, len(buffer) - offset)} remain")


# --------------------------------------------------------------------------
# HDF5
# --------------------------------------------------------------------------

def import_h5py():
    """Import h5py on demand, turning a missing library into a clear message."""
    try:
        import h5py
    except ImportError as exc:
        raise DependencyError(
            "HDF5 support requires the 'h5py' package, which is not installed. "
            "Install it with: pip install h5py  (bin and csv schemas work without it)"
        ) from exc
    return h5py


class Hdf5Reader:
    """Reads datasets and attributes from an HDF5 file by path."""

    def __init__(self, schema: Schema, path: str, needed: set[str]):
        self.schema = schema
        self.path = path
        self.needed = needed

    def iter_results(self, wanted: int | None) -> Iterator[tuple[int, dict[str, Any]]]:
        """Yield one (index, values) pair per matching frame index."""
        h5py = import_h5py()
        try:
            handle = h5py.File(self.path, "r")
        except OSError as exc:
            raise DataError(f"{self.path}: cannot open as HDF5: {exc}") from exc
        with handle:
            indices = self._indices(handle, wanted)
            log.info("%s: %d index value(s): %s", self.path, len(indices),
                     ", ".join(str(item) for item in indices[:10])
                     + (" ..." if len(indices) > 10 else ""))
            for index in indices:
                yield index, self._read(handle, index)

    def _indices(self, handle, wanted: int | None) -> list[int]:
        """Discover the frame indices present, or confirm the requested one."""
        if not self.schema.index_dependent:
            return [0]
        if wanted is not None:
            return [wanted]
        template = next(field.spec["object"] for field in self.schema.index.values()
                        if field.kind != "pseudo" and "{index}" in field.spec.get("at", ""))
        pattern = re.escape(template).replace(re.escape("{index}"), r"(\d+)")
        matcher = re.compile("^" + pattern + "$")
        found: set[int] = set()
        catalog = ["/"]
        handle.visit(lambda name: catalog.append("/" + name))
        for candidate in catalog:
            match = matcher.match(candidate)
            if match:
                found.add(int(match.group(1)))
        if not found:
            raise DataError(f"{self.path}: no object matches {template}")
        return sorted(found)

    def _read(self, handle, index: int) -> dict[str, Any]:
        """Read every needed field at one index."""
        values: dict[str, Any] = {}
        for field in sorted(self.schema.index.values(), key=lambda item: item.order):
            location = field.spec["at"].replace("{index}", str(index))
            object_path = field.spec["object"].replace("{index}", str(index))
            attribute = field.spec["attribute"]
            if field.path not in self.needed:
                trace(f"{self.path}: {field.path} skipped (not in output)")
                continue
            if object_path not in handle:
                raise DataError(f"{self.path}: {object_path} not found (for field {field.path})")
            node = handle[object_path]
            if attribute is not None:
                if attribute not in node.attrs:
                    raise DataError(f"{self.path}: attribute {attribute!r} not found "
                                    f"on {object_path}")
                raw = node.attrs[attribute]
            else:
                if not hasattr(node, "shape"):
                    raise DataError(f"{self.path}: {object_path} is a group, not a dataset")
                if field.slice is not None:
                    # Slicing the dataset rather than the result keeps a large
                    # frame from being read in full just to be decimated.
                    try:
                        raw = node[field.slice]
                    except (IndexError, ValueError, TypeError) as exc:
                        raise DataError(f"{self.path}: {location}: slice cannot be applied "
                                        f"to shape {list(node.shape)}: {exc}") from exc
                else:
                    raw = node[()]
            values[field.path] = self._convert(field, raw, location, values)
            trace(f"{self.path}: {field.path} from {location}")
        return values

    def _convert(self, field: Field, raw: Any, location: str, values: dict) -> Any:
        """Validate a value against any declared type and shape, and normalise it."""
        array = np.asarray(raw) if not isinstance(raw, (str, bytes)) else None
        # An attribute is normally a single value, but nothing in HDF5 prevents
        # one from holding a vector, so the value decides rather than the schema.
        if array is None or array.ndim == 0:
            value = jsonable(raw if array is None else array[()])
            self._check_type(field, value, location)
            self._check_expect(field, value, location)
            return self._apply_enum(field, value, location)
        if field.type is not None:
            expected = numpy_dtype(field.type)
            if array.dtype.kind != expected.kind or array.dtype.itemsize != expected.itemsize:
                raise DataError(f"{self.path}: {location} holds {array.dtype}, "
                                f"schema declares {field.type}")
        if field.shape is not None:
            dims = [item.as_int(values, f"field {field.path} shape") for item in field.shape]
            if list(array.shape) != dims:
                raise DataError(f"{self.path}: {location} has shape {list(array.shape)}, "
                                f"schema declares {dims}")
        return array

    def _check_expect(self, field: Field, value: Any, location: str) -> None:
        """Compare a value against the field's declared constant."""
        if field.expect is not None and value != field.expect:
            raise DataError(f"{self.path}: {location}: expected {field.expect!r}, "
                            f"found {value!r}")

    def _apply_enum(self, field: Field, value: Any, location: str) -> Any:
        """Map a raw value through the field's enum table, if it has one."""
        if field.enum is None:
            return value
        if value in field.enum:
            return field.enum[value]
        raise DataError(f"{self.path}: {location}: value {value!r} is not in the enum table")

    def _check_type(self, field: Field, value: Any, location: str) -> None:
        """Check a scalar against its declared type, where one was given."""
        if field.type is None:
            return
        if field.type == "str":
            if not isinstance(value, str):
                raise DataError(f"{self.path}: {location} holds {type(value).__name__}, "
                                f"schema declares str")
        elif isinstance(value, str):
            raise DataError(f"{self.path}: {location} holds text, "
                            f"schema declares {field.type}")


# --------------------------------------------------------------------------
# CSV
# --------------------------------------------------------------------------

class CsvReader:
    """Reads cells, whole columns and column groups out of a delimited text file."""

    def __init__(self, schema: Schema, path: str, needed: set[str]):
        self.schema = schema
        self.path = path
        self.needed = needed

    def iter_results(self, wanted: int | None) -> Iterator[tuple[int, dict[str, Any]]]:
        """Yield one (index, values) pair per data row, or one pair overall."""
        options = self.schema.csv_options
        try:
            # utf-8-sig strips a byte-order mark, which would otherwise end up
            # inside the first column name and break every lookup on it.
            with open(self.path, newline="", encoding="utf-8-sig") as handle:
                rows = list(csvmod.reader(handle, delimiter=options["delimiter"]))
        except OSError as exc:
            raise DataError(f"{self.path}: cannot read file: {exc}") from exc
        except UnicodeDecodeError as exc:
            raise DataError(f"{self.path}: not valid UTF-8 text: {exc}") from exc

        rows = rows[options["skip_lines"]:]
        header: list[str] = []
        if options["header"]:
            if not rows:
                raise DataError(f"{self.path}: no header row present")
            header = [cell.strip() for cell in rows[0]]
            rows = rows[1:]
        rows = [row for row in rows if row and any(cell.strip() for cell in row)]
        log.info("%s: %d data row(s), %d column(s)", self.path, len(rows),
                 len(header) if header else (len(rows[0]) if rows else 0))

        positions = {field.path: [self._column(column, header, field.path)
                                  for column in field.spec["columns"]]
                     for field in self.schema.index.values() if field.kind != "pseudo"}
        base: dict[str, Any] = {"nrows": len(rows)}
        indices = ([wanted] if wanted is not None
                   else (list(range(len(rows))) if self.schema.index_dependent else [0]))
        if not rows and self.schema.index_dependent and wanted is None:
            raise DataError(f"{self.path}: no data rows")
        for index in indices:
            values = dict(base)
            for field in sorted(self.schema.index.values(), key=lambda item: item.order):
                if field.kind == "pseudo" or field.path not in self.needed:
                    continue
                values[field.path] = self._value(field, positions[field.path], rows, index, values)
            yield index, values

    def _column(self, column: Any, header: list[str], where: str) -> int:
        """Resolve a column reference to a zero-based position."""
        if isinstance(column, int):
            return column
        if not header:
            raise DataError(f"{self.path}: field {where} names column {column!r} but the "
                            f"schema declares header: false")
        if column not in header:
            raise DataError(f"{self.path}: column {column!r} not found "
                            f"(present: {', '.join(header)})")
        return header.index(column)

    def _cell(self, rows: list[list[str]], row: int, position: int, where: str) -> str:
        """Fetch one cell, treating short rows as empty cells."""
        if row >= len(rows):
            raise DataError(f"{self.path}: {where}: row {row} does not exist "
                            f"({len(rows)} data row(s))")
        line = rows[row]
        if position >= len(line):
            log.warning("%s: row %d has %d column(s), %s needs column %d; treated as empty",
                        self.path, row, len(line), where, position)
            return ""
        return line[position]

    @staticmethod
    def _fill(value: Any, type_name: str, where: str) -> Any:
        """Substitute a representable placeholder for an empty cell in an array."""
        # Text cells are never empty-as-missing: they read back as "".
        if value is not None:
            return value
        if type_name.startswith("f"):
            return float("nan")
        raise DataError(f"{where}: empty cell cannot be held by an array of {type_name}; "
                        f"use type f32 or f64, which represent it as null")

    def _value(self, field: Field, positions: list[int], rows: list[list[str]],
               index: int, values: dict) -> Any:
        """Read one field's value at the given row index."""
        assert field.type is not None
        if field.kind == "scalar":
            row = field.spec["row"] if field.spec["row"] is not None else index
            text = self._cell(rows, row, positions[0], field.path)
            where = f"{self.path}: row {row}, field {field.path}"
            value = convert_text(text, field.type, where)
            if field.expect is not None and value != field.expect:
                raise DataError(f"{where}: expected {field.expect!r}, found {value!r}")
            if field.enum is not None:
                if value not in field.enum:
                    raise DataError(f"{where}: value {value!r} is not in the enum table")
                value = field.enum[value]
            trace(f"{self.path}: {field.path} = {value!r} (row {row})")
            return value
        cells = [[self._fill(convert_text(self._cell(rows, row, position, field.path),
                                          field.type,
                                          f"{self.path}: row {row}, field {field.path}"),
                             field.type, f"{self.path}: row {row}, field {field.path}")
                  for position in positions]
                 for row in range(len(rows))]
        dtype = numpy_dtype(field.type) if field.type != "str" else np.dtype(object)
        flat = cells if field.spec["multi"] else [row[0] for row in cells]
        array = np.array(flat, dtype=dtype)
        if field.shape is not None:
            dims = [item.as_int(values, f"field {field.path} shape") for item in field.shape]
            if list(array.shape) != dims:
                raise DataError(f"{self.path}: field {field.path} has shape "
                                f"{list(array.shape)}, schema declares {dims}")
        array = apply_slice(array, field.slice, f"{self.path}: field {field.path}")
        trace(f"{self.path}: {field.path} shape {list(array.shape)}")
        return array


READERS: dict[str, Any] = {"bin": BinReader, "hdf5": Hdf5Reader, "csv": CsvReader}
