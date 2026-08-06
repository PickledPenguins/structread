#!/usr/bin/env python3
"""structread - extract values and arrays from binary, HDF5 and CSV files.

This is the front end: it parses the command line, drives one run across every
``-s`` group, and turns every failure into an exit code. The work itself lives
in the modules beside it:

    core.py         errors, logging, the type table, value conversion
    expressions.py  the expression engine and the slice parser
    schema.py       the schema model and its validation
    readers.py      one reader per input format
    writers.py      one writer per output format

The public names those modules define are re-exported here, so ``import
structread`` is enough to reach the whole tool.

See README.md for the schema reference and usage examples.
"""

from __future__ import annotations

import argparse
import csv as csvmod
import os
import sys
from typing import Any

import numpy as np

from core import (
    ARRAY_OUTPUTS,
    BIN_DIRECTIVES,
    BIN_FIELD_KEYS,
    BIN_INPUT_KEYS,
    BIN_STRUCT_KEYS,
    BIN_TYPES,
    CSV_FIELD_KEYS,
    CSV_INPUT_KEYS,
    ENDIAN_CODES,
    EXIT_DATA,
    EXIT_DEPENDENCY,
    EXIT_OK,
    EXIT_SCHEMA,
    EXIT_USAGE,
    HDF5_FIELD_KEYS,
    HDF5_INPUT_KEYS,
    INPUT_FORMATS,
    NUMERIC_TYPES,
    OUTPUT_FIELD_KEYS,
    OUTPUT_FORMATS,
    OUTPUT_KEYS,
    RESERVED_FIELDS,
    SCHEMA_KEYS_OPTIONAL,
    SCHEMA_KEYS_REQUIRED,
    TEXT_TYPES,
    TRACE,
    DataError,
    DependencyError,
    OutputError,
    SchemaError,
    StructReadError,
    UsageError,
    __version__,
    jsonable,
    log,
    numpy_dtype,
    setup_logging,
    trace,
    type_size,
)
from expressions import Expression, apply_slice, compile_expression, parse_slice
from readers import READERS, BinReader, CsvReader, Hdf5Reader, import_h5py
from schema import Field, OutputSpec, Schema, load_schema
from writers import BinWriter, CsvWriter, JsonWriter, describe_array, make_writer, render_value

# Re-exported so that ``import structread`` reaches the whole tool. The
# entry point is a thin front end; these names live in the modules above.
__all__ = [
    "ARRAY_OUTPUTS",
    "BIN_DIRECTIVES",
    "BIN_FIELD_KEYS",
    "BIN_INPUT_KEYS",
    "BIN_STRUCT_KEYS",
    "BIN_TYPES",
    "CSV_FIELD_KEYS",
    "CSV_INPUT_KEYS",
    "ENDIAN_CODES",
    "EXIT_DATA",
    "EXIT_DEPENDENCY",
    "EXIT_OK",
    "EXIT_SCHEMA",
    "EXIT_USAGE",
    "HDF5_FIELD_KEYS",
    "HDF5_INPUT_KEYS",
    "INPUT_FORMATS",
    "NUMERIC_TYPES",
    "OUTPUT_FIELD_KEYS",
    "OUTPUT_FORMATS",
    "OUTPUT_KEYS",
    "READERS",
    "RESERVED_FIELDS",
    "SCHEMA_KEYS_OPTIONAL",
    "SCHEMA_KEYS_REQUIRED",
    "TEXT_TYPES",
    "TRACE",
    "BinReader",
    "BinWriter",
    "CsvReader",
    "CsvWriter",
    "DataError",
    "DependencyError",
    "Expression",
    "Field",
    "Hdf5Reader",
    "JsonWriter",
    "OutputError",
    "OutputSpec",
    "Schema",
    "SchemaError",
    "StructReadError",
    "UsageError",
    "__version__",
    "apply_slice",
    "compile_expression",
    "describe",
    "describe_array",
    "execute",
    "import_h5py",
    "inspect",
    "jsonable",
    "load_schema",
    "log",
    "main",
    "make_writer",
    "needed_fields",
    "numpy_dtype",
    "parse_index",
    "parse_slice",
    "render_value",
    "setup_logging",
    "trace",
    "type_size",
]


# --------------------------------------------------------------------------
# Run orchestration
# --------------------------------------------------------------------------

def needed_fields(schema: Schema) -> set[str]:
    """Every field that must be read: the output fields plus their dependencies."""
    needed = {target for target, _ in schema.output.fields if target not in RESERVED_FIELDS}
    if schema.record_count is not None:
        needed.update(schema.record_count.refs)
    pending = list(needed)
    while pending:
        field = schema.index.get(pending.pop())
        if field is None or field.shape is None:
            continue
        for expression in field.shape:
            for reference in expression.refs:
                if reference not in needed:
                    needed.add(reference)
                    pending.append(reference)
    return needed


def check_agreement(schemas: list[Schema], override_format: str | None) -> None:
    """Verify that every schema in a run describes the same output structure."""
    first = schemas[0]
    for other in schemas[1:]:
        if override_format is None and other.output.format != first.output.format:
            raise SchemaError(
                f"schema mismatch: {first.name!r} outputs {first.output.format}, "
                f"{other.name!r} outputs {other.output.format}; all schemas in one run "
                f"must agree, or use --format")
        if [name for _, name in other.output.fields] != [name for _, name in first.output.fields]:
            raise SchemaError(
                f"schema mismatch: {first.name!r} emits "
                f"[{', '.join(name for _, name in first.output.fields)}] but {other.name!r} "
                f"emits [{', '.join(name for _, name in other.output.fields)}]; "
                f"all schemas in one run must emit the same fields in the same order")


def resolve_destination(schemas: list[Schema], override: str | None) -> str | None:
    """Choose the output path from the command line or the schemas."""
    if override is not None:
        return None if override == "-" else override
    declared = {schema.output.path for schema in schemas if schema.output.path}
    if len(declared) > 1:
        raise SchemaError(f"schemas declare conflicting output paths "
                          f"({', '.join(sorted(declared))}); use -o to choose one")
    return declared.pop() if declared else None


def summarise(schemas: list[Schema], groups: list[tuple[Schema, list[str]]],
              fmt: str, path: str | None, index: int | None) -> None:
    """Log the startup summary: what will be read, how, and where it will go."""
    log.info("structread %s", __version__)
    for schema, files in groups:
        title = f"{schema.name}"
        if schema.version is not None:
            title += f" v{schema.version}"
        log.info("schema    %s (%s)", title, schema.file)
        if schema.source:
            log.info("          source: %s", schema.source)
        if schema.description:
            log.info("          %s", str(schema.description).strip().splitlines()[0])
        if schema.format == "bin":
            log.info("input     bin, endian=%s", schema.endian_name)
            for group in ("header", "record"):
                field = schema.index.get(group)
                if field is not None:
                    log.info("          %s: %d field(s)", group, len(field.children))
            if schema.record_count is not None:
                log.info("          record_count: %s", schema.record_count.source)
        else:
            log.info("input     %s, %d field(s)", schema.format, len(schema.leaf_paths()))
        log.info("files     %s", ", ".join(files) if files else "(none)")
    log.info("index     %s%s", "all" if index is None else index,
             " [default]" if index is None else "")
    log.info("output    %s -> %s", fmt, path or "standard output")
    log.info("          fields: %s", ", ".join(name for _, name in schemas[0].output.fields))
    for schema in schemas:
        wanted = needed_fields(schema)
        skipped = [item for item in schema.leaf_paths() if item not in wanted]
        for note in schema.assumptions:
            log.info("assumed   [%s] %s", schema.name, note)
        if skipped:
            log.info("assumed   [%s] not in output, will not be read: %s",
                     schema.name, ", ".join(skipped))


def build_record(schema: Schema, values: dict[str, Any], context: dict) -> dict[str, Any]:
    """Assemble one output record from the values read and the reserved fields."""
    record: dict[str, Any] = {}
    for target, out_name in schema.output.fields:
        if target in RESERVED_FIELDS:
            record[out_name] = context[target]
        else:
            record[out_name] = values.get(target)
    return record


def execute(args: argparse.Namespace, groups: list[tuple[Schema, list[str]]], stdout) -> int:
    """Read every file in every group and write the combined output."""
    schemas = [schema for schema, _ in groups]
    check_agreement(schemas, args.format)
    fmt = args.format or schemas[0].output.format
    if args.format:
        for schema in schemas:
            if fmt == "csv":
                arrays = [target for target, _ in schema.output.fields
                          if target not in RESERVED_FIELDS
                          and (schema.index[target].is_array
                               or schema.index[target].kind == "list"
                               or schema.inside_list(target) is not None)]
                if arrays:
                    raise UsageError(f"--format csv cannot hold multi-valued field(s) "
                                     f"{', '.join(arrays)} from schema {schema.name!r}")
            if fmt in ARRAY_OUTPUTS and len(schema.output.fields) != 1:
                raise UsageError(f"--format {fmt} takes exactly one array field, but schema "
                                 f"{schema.name!r} emits {len(schema.output.fields)}")
    path = resolve_destination(schemas, args.output)
    summarise(schemas, groups, fmt, path, args.index)
    writer = make_writer(fmt, path, schemas, args.indent)

    failures = 0
    for schema, files in groups:
        needed = needed_fields(schema)
        for source in files:
            log.debug("reading %s with schema %r", source, schema.name)
            stem = os.path.splitext(os.path.basename(source))[0]
            try:
                if not os.path.exists(source):
                    raise DataError(f"{source}: file not found")
                reader = READERS[schema.format](schema, source, needed)
                for index, values in reader.iter_results(args.index):
                    context = {"source": source, "stem": stem, "index": index,
                               "schema": schema.name}
                    writer.write(build_record(schema, values, context), context)
            except (DataError, OutputError) as exc:
                if not args.skip_errors:
                    raise
                failures += 1
                log.warning("%s; file skipped", exc)
    written = writer.close(stdout)
    log.info("wrote %d %s to %s", written,
             "array(s)" if fmt in ARRAY_OUTPUTS else "record(s)", path or "standard output")
    if failures:
        log.warning("%d file(s) skipped", failures)
    return EXIT_DATA if failures else EXIT_OK


# --------------------------------------------------------------------------
# Describing a schema against a file, and inspecting a file without one
# --------------------------------------------------------------------------

def _cell_text(value: Any, width: int = 34) -> str:
    """Render a value for a describe table, shortened if it is long."""
    if isinstance(value, np.ndarray):
        text = f"<{value.dtype} array {list(value.shape)}>"
    elif isinstance(value, list):
        text = f"<{len(value)} item(s)>"
    else:
        text = repr(jsonable(value))
    return text if len(text) <= width else text[: width - 1] + "…"


def _table(rows: list[tuple], headings: tuple) -> list[str]:
    """Format rows as a fixed-width table."""
    columns = list(zip(*([headings, *rows]), strict=True)) if rows else [(item,) for item
                                                                         in headings]
    widths = [max(len(str(cell)) for cell in column) for column in columns]
    lines = ["  ".join(str(cell).ljust(width) for cell, width in zip(headings, widths,
                                                                    strict=True)).rstrip(),
             "  ".join("-" * width for width in widths)]
    lines += ["  ".join(str(cell).ljust(width) for cell, width in zip(row, widths,
                                                                     strict=True)).rstrip()
              for row in rows]
    return lines


def describe(schema: Schema, path: str, index: int | None, out) -> None:
    """Print how a schema resolves against one file: locations, sizes and values."""
    wanted = 0 if index is None else index
    reader = READERS[schema.format](schema, path, set(schema.index))
    values: dict[str, Any] = {}
    for _, produced in reader.iter_results(wanted):
        values = produced
        break
    out.write(f"{path}  [schema {schema.name}, index {wanted}]\n")
    rows: list[tuple] = []
    headings: tuple
    if schema.format == "bin":
        offsets = {item[0]: (item[1], item[2]) for item in reader.layout}
        rows += [(field.path, field.type or field.kind,
                 offsets.get(field.path, ("", ""))[0], offsets.get(field.path, ("", ""))[1],
                 _cell_text(values.get(field.path)))
                for field in sorted(schema.index.values(), key=lambda item: item.order)
                if field.is_value]
        headings = ("field", "type", "offset", "bytes", "value")
    elif schema.format == "hdf5":
        rows += [(field.path, field.spec["at"].replace("{index}", str(wanted)),
                 field.type or "(from file)", _cell_text(values.get(field.path)))
                for field in sorted(schema.index.values(), key=lambda item: item.order)
                if field.is_value]
        headings = ("field", "location", "type", "value")
    else:
        rows += [(field.path, ", ".join(str(item) for item in field.spec["columns"]),
                 field.type or "str",
                 "yes" if field.spec.get("row") is not None else "no",
                 _cell_text(values.get(field.path)))
                for field in sorted(schema.index.values(), key=lambda item: item.order)
                if field.is_value]
        headings = ("field", "column", "type", "pinned", "value")
    for line in _table(rows, headings):
        out.write(line + "\n")
    out.write("\n")


def guess_format(path: str) -> str:
    """Guess an input format from a filename, for inspection without a schema."""
    suffix = os.path.splitext(path)[1].lower()
    if suffix in (".h5", ".hdf5", ".he5"):
        return "hdf5"
    if suffix in (".csv", ".tsv", ".txt"):
        return "csv"
    return "bin"


def inspect(path: str, hint: str | None, out) -> None:
    """Print what is in a file, to help write a schema for it."""
    fmt = hint or guess_format(path)
    size = os.path.getsize(path)
    out.write(f"{path}  [{fmt}, {size} byte(s)]\n")
    if fmt == "hdf5":
        _inspect_hdf5(path, out)
    elif fmt == "csv":
        _inspect_csv(path, out)
    else:
        _inspect_bin(path, out)
    out.write("\n")


def _inspect_hdf5(path: str, out) -> None:
    """List every object and attribute in an HDF5 file."""
    h5py = import_h5py()
    try:
        handle = h5py.File(path, "r")
    except OSError as exc:
        raise DataError(f"{path}: cannot open as HDF5: {exc}") from exc
    rows: list[tuple] = []

    def record(name: str, node) -> None:
        """Add one object, then its attributes, to the listing."""
        location = "/" + name
        if hasattr(node, "shape"):
            rows.append((location, "dataset", str(node.dtype), str(list(node.shape))))
        else:
            rows.append((location, "group", "", ""))
        for key in node.attrs:
            rows.append((f"{location}@{key}", "attribute",
                         type(jsonable(node.attrs[key])).__name__,
                         _cell_text(node.attrs[key], 24)))

    with handle:
        for key in handle.attrs:
            rows.append((f"/@{key}", "attribute", type(jsonable(handle.attrs[key])).__name__,
                         _cell_text(handle.attrs[key], 24)))
        handle.visititems(record)
    for line in _table(rows, ("location", "kind", "type", "shape or value")):
        out.write(line + "\n")


def _inspect_csv(path: str, out) -> None:
    """Show the delimiter, the columns and the first rows of a delimited file."""
    try:
        with open(path, newline="", encoding="utf-8-sig", errors="replace") as handle:
            sample = handle.read(8192)
            handle.seek(0)
            delimiter = max(",;\t|", key=sample.count)
            rows = []
            for position, row in enumerate(csvmod.reader(handle, delimiter=delimiter)):
                rows.append(row)
                if position >= 5:
                    break
    except OSError as exc:
        raise DataError(f"{path}: cannot read file: {exc}") from exc
    out.write(f"delimiter {delimiter!r} (guessed from the first 8 kB)\n")
    if not rows:
        out.write("(no rows)\n")
        return
    listing = [(str(position), name, _cell_text(rows[1][position] if len(rows) > 1
                                                and position < len(rows[1]) else "", 24))
               for position, name in enumerate(rows[0])]
    for line in _table(listing, ("column", "first row", "second row")):
        out.write(line + "\n")


def _inspect_bin(path: str, out) -> None:
    """Show the opening bytes of a file as hex and text."""
    with open(path, "rb") as handle:
        head = handle.read(96)
    for start in range(0, len(head), 16):
        chunk = head[start:start + 16]
        hexed = " ".join(f"{byte:02x}" for byte in chunk)
        text = "".join(chr(byte) if 32 <= byte < 127 else "." for byte in chunk)
        out.write(f"{start:08d}  {hexed:<47}  {text}\n")


# --------------------------------------------------------------------------
# Command line interface
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="structread",
        usage="structread [-h] -s SCHEMA INPUT... [-s SCHEMA INPUT...] [options]\n"
              "       structread --inspect FILE... [--inspect-as FORMAT]",
        description="Extract values and arrays from binary, HDF5 and CSV files "
                    "using YAML schemas.",
        epilog="Exit status: 0 success, 1 data error, 2 schema error, "
               "3 usage error, 4 missing dependency.",
    )
    parser.add_argument("-s", "--schema", action="append", nargs="+",
                        metavar="SCHEMA INPUT", dest="groups",
                        help="a schema followed by the input files it reads; repeatable")
    parser.add_argument("--inspect", nargs="+", metavar="FILE",
                        help="describe what is in these files without a schema, then exit")
    parser.add_argument("--inspect-as", choices=("bin", "hdf5", "csv"), metavar="FORMAT",
                        help="treat --inspect files as this format instead of guessing")
    parser.add_argument("-o", "--output", metavar="PATH",
                        help="destination; '-' means standard output. May contain "
                             "{stem} and {index} for bin output")
    parser.add_argument("--format", choices=OUTPUT_FORMATS,
                        help="override the output format of every schema")
    parser.add_argument("--index", default="all", metavar="SPEC",
                        help="record, frame or row to read: an integer or 'all' (default: all)")
    parser.add_argument("--indent", type=int, default=2, metavar="N",
                        help="JSON indentation, 0 for compact (default: 2)")
    parser.add_argument("--skip-errors", action="store_true",
                        help="continue past files that cannot be read")
    parser.add_argument("--validate", action="store_true",
                        help="validate the schemas and exit without reading input")
    parser.add_argument("--describe", action="store_true",
                        help="show how each schema resolves against its input - field "
                             "locations, sizes and values - then exit")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="log warnings and errors only")
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="log at debug level; repeat for per-field tracing")
    parser.add_argument("--log-file", metavar="PATH",
                        help="also write a debug-level log to this file")
    parser.add_argument("--version", action="version", version=f"structread {__version__}")
    return parser


def parse_index(text: str) -> int | None:
    """Interpret the --index argument, where None means every index."""
    if text == "all":
        return None
    try:
        value = int(text)
    except ValueError as exc:
        raise UsageError(f"--index: expected an integer or 'all', got {text!r}") from exc
    if value < 0:
        raise UsageError("--index: expected a non-negative integer or 'all'")
    return value


def collect_groups(raw_groups: list[list[str]], validate_only: bool) -> list[tuple[Schema, list]]:
    """Load each schema and pair it with the input files that follow it."""
    groups = []
    for entry in raw_groups:
        schema_path, files = entry[0], entry[1:]
        if not files and not validate_only:
            raise UsageError(f"-s {schema_path}: no input files given for this schema")
        groups.append((load_schema(schema_path), files))
    return groups


def main(argv: list[str] | None = None, stdout=None, stderr=None) -> int:
    """Entry point: parse arguments, run, and map failures to exit codes."""
    stdout = stdout if stdout is not None else sys.stdout
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        setup_logging(args.quiet, args.verbose, args.log_file)
        if args.indent < 0:
            raise UsageError("--indent must be 0 or greater")
        args.index = parse_index(args.index)
        if args.inspect:
            for source in args.inspect:
                if not os.path.exists(source):
                    raise DataError(f"{source}: file not found")
                inspect(source, args.inspect_as, stdout)
            return EXIT_OK
        if not args.groups:
            raise UsageError("at least one -s SCHEMA INPUT... group is required "
                             "(or --inspect FILE)")
        groups = collect_groups(args.groups, args.validate)
        if args.describe:
            for schema, files in groups:
                for source in files:
                    if not os.path.exists(source):
                        raise DataError(f"{source}: file not found")
                    describe(schema, source, args.index, stdout)
            return EXIT_OK
        if args.validate:
            for schema, _ in groups:
                log.info("%s: schema %r is valid (%d field(s), output %s)",
                         schema.file, schema.name, len(schema.leaf_paths()),
                         schema.output.format)
            check_agreement([schema for schema, _ in groups], args.format)
            return EXIT_OK
        return execute(args, groups, stdout)
    except SchemaError as exc:
        log.error("schema error: %s", exc)
        return EXIT_SCHEMA
    except DependencyError as exc:
        log.error("%s", exc)
        return EXIT_DEPENDENCY
    except UsageError as exc:
        log.error("usage error: %s", exc)
        return EXIT_USAGE
    except (DataError, OutputError) as exc:
        log.error("%s", exc)
        return EXIT_DATA
    except BrokenPipeError:
        # The consumer of a piped stream stopped reading; this is not a failure.
        # The real standard output is redirected so that interpreter shutdown
        # does not report the same broken pipe a second time. An injected
        # stream is left alone: it belongs to the caller.
        if stdout is sys.stdout:
            try:
                os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
            except (OSError, ValueError):
                pass
        return EXIT_OK
    except OSError as exc:
        log.error("%s", exc)
        return EXIT_DATA
    except KeyboardInterrupt:
        log.error("interrupted")
        return EXIT_DATA
    except Exception as exc:  # noqa: BLE001 - a CLI must never show a traceback
        log.error("unexpected %s while reading input: %s", type(exc).__name__, exc)
        log.debug("details", exc_info=True)
        return EXIT_DATA


if __name__ == "__main__":
    sys.exit(main())
