"""The schema model: what a file looks like, and what should come out of it.

A ``Schema`` is built and fully validated before any input file is opened, so
a malformed schema costs nothing to detect and a run that starts will not stop
for a schema reason. Validation is deliberately strict - an unrecognised key is
an error rather than being ignored, and a name that could resolve two ways is
an error rather than being resolved silently.

The central type is ``Field``. Every node in a schema is one, and its ``kind``
decides how the readers and writers treat it:

    scalar   one number, boolean or string
    string   a char array decoded to text
    array    a numpy array
    list     a repeating structure, read as a list of items
    struct   a non-repeating structure; contributes only its members
    pad      skip forward a fixed number of bytes
    seek     move to an absolute offset
    pseudo   a value the reader supplies, currently only CSV's $nrows

``Schema.index`` maps every full path to its ``Field``; ``Field.order`` is a
monotonic counter that makes forward and self references detectable.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Any

import yaml

from core import (
    ARRAY_OUTPUTS,
    BIN_FIELD_KEYS,
    BIN_INPUT_KEYS,
    BIN_TYPES,
    CSV_FIELD_KEYS,
    CSV_INPUT_KEYS,
    ENDIAN_CODES,
    HDF5_FIELD_KEYS,
    HDF5_INPUT_KEYS,
    INPUT_FORMATS,
    NAME_RE,
    OUTPUT_FORMATS,
    OUTPUT_KEYS,
    RESERVED_FIELDS,
    SCHEMA_KEYS_OPTIONAL,
    SCHEMA_KEYS_REQUIRED,
    TEXT_TYPES,
    SchemaError,
    log,
)
from expressions import Expression, compile_expression, parse_slice

# --------------------------------------------------------------------------
# The field tree
# --------------------------------------------------------------------------

class Field:
    """One field in a schema, whether a scalar, an array, or a nested structure."""

    def __init__(self, path: str, name: str, kind: str, spec: dict, order: int):
        self.path = path
        self.name = name
        self.kind = kind          # scalar | string | array | struct | pad
        self.spec = spec
        self.order = order
        self.type: str | None = None
        self.shape: list[Expression] | None = None
        self.children: list[Field] = []
        self.expect: Any = None
        self.enum: dict | None = None
        self.slice: tuple | None = None
        self.count: Expression | None = None   # set on a repeating structure

    def __repr__(self) -> str:
        """Identify the field by path and kind in diagnostics."""
        return f"Field({self.path}, {self.kind})"

    @property
    def is_array(self) -> bool:
        """True when this field yields an array rather than a single value."""
        return self.kind == "array"

    @property
    def is_value(self) -> bool:
        """True when this field yields something that can be emitted."""
        return self.kind in ("scalar", "string", "array", "list")


class OutputSpec:
    """The output half of a schema: format, destination and field list."""

    def __init__(self, fmt: str, path: str | None, fields: list[tuple[str, str]]):
        self.format = fmt
        self.path = path
        self.fields = fields      # (field path, output name)

    def signature(self) -> tuple:
        """Comparable form used to check that schemas in one run agree."""
        return (self.format, tuple(self.fields))


# --------------------------------------------------------------------------
# Validation helpers
# --------------------------------------------------------------------------

def _check_keys(mapping: Any, required: Sequence[str], optional: Sequence[str],
                where: str) -> None:
    """Verify a mapping has every required key and no unrecognised ones."""
    if not isinstance(mapping, dict):
        raise SchemaError(f"{where}: expected a mapping, got {type(mapping).__name__}")
    allowed = set(required) | set(optional)
    missing = [key for key in required if key not in mapping]
    if missing:
        raise SchemaError(f"{where}: missing required key(s): {', '.join(sorted(missing))}")
    unknown = [key for key in mapping if key not in allowed]
    if unknown:
        raise SchemaError(f"{where}: unknown key(s): {', '.join(sorted(unknown))} "
                          f"(allowed: {', '.join(sorted(allowed))})")


def _check_name(name: Any, where: str) -> str:
    """Verify a field name is a legal identifier."""
    if not isinstance(name, str) or not NAME_RE.match(name):
        raise SchemaError(
            f"{where}: {name!r} is not a usable field name; names must start with a letter "
            f"or underscore and contain only letters, digits and underscores - "
            f"add an explicit 'name:' if this came from a column or attribute"
        )
    if name in RESERVED_FIELDS:
        raise SchemaError(f"{where}: {name!r} is reserved and cannot be used as a field name")
    return name


def _check_enum(entry: dict, where: str) -> dict | None:
    """Validate an enum table: a mapping of raw value to name."""
    if "enum" not in entry:
        return None
    table = entry["enum"]
    if not isinstance(table, dict) or not table:
        raise SchemaError(f"{where}.enum: expected a non-empty mapping of value to name")
    for key, value in table.items():
        if isinstance(key, bool) or not isinstance(key, (int, str)):
            raise SchemaError(f"{where}.enum: {key!r} is not a usable raw value")
        if not isinstance(value, str) or not value:
            raise SchemaError(f"{where}.enum[{key!r}]: expected a non-empty name")
    return table


def _check_slice(entry: dict, where: str) -> tuple | None:
    """Validate and parse a numpy-style slice string."""
    if "slice" not in entry:
        return None
    return parse_slice(_check_str(entry["slice"], f"{where}.slice"))


# --------------------------------------------------------------------------
# The schema
# --------------------------------------------------------------------------

class Schema:
    """A validated schema: its input structure, its output structure, and metadata."""

    def __init__(self, document: dict, path: str):
        self.file = path
        _check_keys(document, SCHEMA_KEYS_REQUIRED, SCHEMA_KEYS_OPTIONAL, "schema")
        self.name = _check_str(document["name"], "schema.name")
        self.version = document.get("version")
        self.source = document.get("source")
        self.description = document.get("description")
        self.aliases = self._read_aliases(document.get("aliases"))
        self.assumptions: list[str] = []

        block = document["input"]
        if not isinstance(block, dict) or "format" not in block:
            raise SchemaError("schema.input: expected a mapping with a 'format' key")
        self.format = block["format"]
        if self.format not in INPUT_FORMATS:
            raise SchemaError(f"schema.input.format: expected bin, hdf5 or csv, "
                              f"got {self.format!r}")

        self.index: dict[str, Field] = {}
        self.roots: list[Field] = []
        self._order = 0
        self.endian = "<"
        self.record_count: Expression | None = None
        self.record_size: Expression | None = None
        self.index_dependent = False
        self.csv_options: dict[str, Any] = {}

        builders = {"bin": self._build_bin, "hdf5": self._build_hdf5, "csv": self._build_csv}
        builders[self.format](block)
        self._check_alias_collisions()
        self.output = self._build_output(document.get("output"))

    # -- schema header ----------------------------------------------------

    def _read_aliases(self, raw: Any) -> dict[str, str]:
        """Validate the alias table: no shadowing, no chaining, all strings."""
        if raw is None:
            return {}
        if not isinstance(raw, dict):
            raise SchemaError("schema.aliases: expected a mapping of short name to full name")
        aliases: dict[str, str] = {}
        for key, value in raw.items():
            if not isinstance(key, str) or not NAME_RE.match(key):
                raise SchemaError(f"schema.aliases: {key!r} is not a usable alias name")
            if not isinstance(value, str) or not value:
                raise SchemaError(f"schema.aliases.{key}: expected a non-empty name")
            if key in RESERVED_FIELDS:
                raise SchemaError(f"schema.aliases.{key}: shadows the reserved field {key!r}")
            aliases[key] = value
        for key, value in aliases.items():
            if value.split(".")[0] in aliases:
                raise SchemaError(f"schema.aliases.{key}: {value!r} chains to another alias, "
                                  f"which is not allowed")
        return aliases

    def _check_alias_collisions(self) -> None:
        """Reject an alias that would shadow a real field.

        Aliases are expanded before references are resolved, so an alias whose
        name matches a field would win silently - the opposite of the ambiguity
        rule that governs every other name. An alias that names the very field
        it points at is harmless and stays allowed, since it only widens where
        that field can be reached from.
        """
        # Padding entries are never registered, so every indexed field has a name.
        known: dict[str, set[str]] = {}
        for path, field in self.index.items():
            known.setdefault(path, set()).add(path)
            known.setdefault(field.name, set()).add(path)
        for key, value in self.aliases.items():
            shadowed = known.get(key)
            if shadowed and shadowed != {value}:
                raise SchemaError(
                    f"schema.aliases.{key}: shadows the field(s) "
                    f"{', '.join(sorted(shadowed))}; choose an alias name that is not "
                    f"a field name, or point it at that field")

    def expand_alias(self, name: str) -> str:
        """Expand an alias appearing in a reference or a path placeholder."""
        if name in self.aliases:
            return self.aliases[name]
        head, _, rest = name.partition(".")
        if head in self.aliases:
            return self.aliases[head] + ("." + rest if rest else "")
        return name

    def expand_placeholders(self, text: str) -> str:
        """Rewrite {alias} placeholders in a path to their full names."""
        def substitute(match: re.Match) -> str:
            """Replace one placeholder with its expanded name."""
            return "{" + self.expand_alias(match.group(1)) + "}"
        return re.sub(r"\{([A-Za-z_][A-Za-z0-9_.]*)\}", substitute, text)

    # -- name resolution --------------------------------------------------

    def _register(self, field: Field) -> Field:
        """Add a field to the path index, rejecting duplicates."""
        if field.path in self.index:
            raise SchemaError(f"duplicate field {field.path!r}")
        self.index[field.path] = field
        return field

    def _next_order(self) -> int:
        """Return the next declaration-order counter."""
        self._order += 1
        return self._order

    def resolver(self, scope: str, limit: int, where: str) -> Callable[[str], str]:
        """Build a reference resolver for one scope and declaration position."""

        def resolve(raw: str) -> str:
            """Resolve one $reference to a full field path."""
            name = self.expand_alias(raw)
            candidates: list[str] = []
            parts = scope.split(".") if scope else []
            while parts:
                candidate = ".".join(parts) + "." + name
                if candidate in self.index:
                    candidates.append(candidate)
                parts = parts[:-1]
            if name in self.index and name not in candidates:
                candidates.append(name)
            if not candidates:
                raise SchemaError(f"{where}: ${raw} does not name any field")
            if len(candidates) > 1:
                raise SchemaError(
                    f"{where}: ${raw} is ambiguous - it could mean "
                    f"{' or '.join(candidates)}; write the full path"
                )
            target = self.index[candidates[0]]
            if target.order >= limit:
                raise SchemaError(f"{where}: ${raw} refers to {target.path}, which is read later")
            if target.kind not in ("scalar", "pseudo"):
                raise SchemaError(f"{where}: ${raw} refers to {target.path}, "
                                  f"which is not a single value")
            return target.path

        return resolve

    def _shape(self, raw: Any, scope: str, limit: int, where: str) -> list[Expression]:
        """Compile a shape list into expressions."""
        if not isinstance(raw, list):
            raise SchemaError(f"{where}.shape: expected a list of dimensions")
        return [compile_expression(item, self.resolver(scope, limit, f"{where}.shape"),
                                   f"{where}.shape") for item in raw]

    # -- binary -----------------------------------------------------------

    def _build_bin(self, block: dict) -> None:
        """Build the field tree for a binary schema."""
        _check_keys(block, ["format"], BIN_INPUT_KEYS, "schema.input")
        endian = block.get("endian", "little")
        if endian not in ENDIAN_CODES:
            raise SchemaError(f"schema.input.endian: expected one of "
                              f"{', '.join(ENDIAN_CODES)}")
        if "endian" not in block:
            self.assumptions.append("endian=little [default]")
        self.endian = ENDIAN_CODES[endian]
        self.endian_name = endian

        if "header" not in block and "record" not in block:
            raise SchemaError("schema.input: a binary schema needs 'header', 'record', or both")

        for group in ("header", "record"):
            if group not in block:
                continue
            entries = block[group]
            if not isinstance(entries, list) or not entries:
                raise SchemaError(f"schema.input.{group}: expected a non-empty list of fields")
            root = self._register(Field(group, group, "struct", {}, self._next_order()))
            self.roots.append(root)
            root.children = self._bin_fields(entries, group, f"schema.input.{group}")

        self.index_dependent = "record" in self.index
        if "record_count" in block:
            if "record" not in self.index:
                raise SchemaError("schema.input.record_count: meaningless without a 'record' block")
            limit = min((field.order for field in self.index.values()
                         if field.path.startswith("record")), default=self._order + 1)
            self.record_count = compile_expression(
                block["record_count"],
                self.resolver("", limit, "schema.input.record_count"),
                "schema.input.record_count")
        elif "record" in self.index:
            self.assumptions.append("record_count not given: records read until end of file")
        if "record_size" in block:
            if "record" not in self.index:
                raise SchemaError("schema.input.record_size: meaningless without a 'record' block")
            limit = min((field.order for field in self.index.values()
                         if field.path.startswith("record")), default=self._order + 1)
            self.record_size = compile_expression(
                block["record_size"],
                self.resolver("", limit, "schema.input.record_size"),
                "schema.input.record_size")

    def _bin_fields(self, entries: list, scope: str, where: str) -> list[Field]:
        """Build one level of binary fields, recursing into nested structures."""
        fields: list[Field] = []
        for position, entry in enumerate(entries):
            spot = f"{where}[{position}]"
            if not isinstance(entry, dict):
                raise SchemaError(f"{spot}: expected a mapping, got {type(entry).__name__}")
            if "pad" in entry:
                _check_keys(entry, ["pad"], [], spot)
                if not isinstance(entry["pad"], int) or isinstance(entry["pad"], bool) \
                        or entry["pad"] < 0:
                    raise SchemaError(f"{spot}.pad: expected a non-negative integer")
                fields.append(Field(f"{scope}.<pad{position}>", "", "pad", entry,
                                    self._next_order()))
                continue
            if "seek" in entry:
                _check_keys(entry, ["seek"], [], spot)
                marker = Field(f"{scope}.<seek{position}>", "", "seek", entry,
                               self._next_order())
                marker.count = compile_expression(
                    entry["seek"], self.resolver(scope, marker.order, f"{spot}.seek"),
                    f"{spot}.seek")
                fields.append(marker)
                continue
            if "fields" in entry:
                _check_keys(entry, ["name", "fields"], ["count"], spot)
                name = _check_name(entry["name"], f"{spot}.name")
                repeated = "count" in entry
                nested = self._register(Field(f"{scope}.{name}", name,
                                              "list" if repeated else "struct", entry,
                                              self._next_order()))
                if repeated:
                    nested.count = compile_expression(
                        entry["count"], self.resolver(scope, nested.order, f"{spot}.count"),
                        f"{spot}.count")
                if not isinstance(entry["fields"], list) or not entry["fields"]:
                    raise SchemaError(f"{spot}.fields: expected a non-empty list")
                nested.children = self._bin_fields(entry["fields"], nested.path, f"{spot}.fields")
                fields.append(nested)
                continue
            _check_keys(entry, ["name", "type"],
                        [key for key in BIN_FIELD_KEYS if key not in ("name", "type")], spot)
            name = _check_name(entry["name"], f"{spot}.name")
            type_name = entry["type"]
            if type_name not in BIN_TYPES:
                raise SchemaError(f"{spot}.type: unknown type {type_name!r} "
                                  f"(known: {', '.join(sorted(BIN_TYPES))})")
            has_shape = "shape" in entry
            if type_name == "char" and not has_shape:
                raise SchemaError(f"{spot}: type char requires a shape, e.g. shape: [16]")
            kind = "string" if type_name == "char" else ("array" if has_shape else "scalar")
            field = self._register(Field(f"{scope}.{name}", name, kind, entry,
                                         self._next_order()))
            field.type = type_name
            if has_shape:
                field.shape = self._shape(entry["shape"], scope, field.order, spot)
            self._value_options(field, entry, spot)
            fields.append(field)
        return fields

    @staticmethod
    def _value_options(field: Field, entry: dict, spot: str) -> None:
        """Attach expect, enum and slice to a value field, checking they apply."""
        field.enum = _check_enum(entry, spot)
        field.slice = _check_slice(entry, spot)
        if field.enum is not None and field.kind != "scalar":
            raise SchemaError(f"{spot}.enum: applies to single values only")
        if field.slice is not None and field.kind != "array":
            raise SchemaError(f"{spot}.slice: applies to array fields only")
        if "expect" in entry:
            if field.kind not in ("scalar", "string"):
                raise SchemaError(f"{spot}.expect: applies to single values only")
            if isinstance(entry["expect"], (dict, list)):
                raise SchemaError(f"{spot}.expect: expected a number or a string")
            field.expect = entry["expect"]

    # -- HDF5 -------------------------------------------------------------

    def _build_hdf5(self, block: dict) -> None:
        """Build the flat field list for an HDF5 schema."""
        _check_keys(block, ["format", "fields"], HDF5_INPUT_KEYS, "schema.input")
        entries = block["fields"]
        if not isinstance(entries, list) or not entries:
            raise SchemaError("schema.input.fields: expected a non-empty list of locations")
        for position, entry in enumerate(entries):
            spot = f"schema.input.fields[{position}]"
            _check_keys(entry, ["at"], [key for key in HDF5_FIELD_KEYS if key != "at"], spot)
            at = self.expand_placeholders(_check_str(entry["at"], f"{spot}.at"))
            object_path, sep, attribute = at.partition("@")
            if "name" in entry:
                name = _check_name(entry["name"], f"{spot}.name")
            else:
                default = attribute if sep else object_path.rstrip("/").rsplit("/", 1)[-1]
                name = _check_name(default, f"{spot}: default name {default!r}")
                self.assumptions.append(
                    f'field "{name}": name from ' + ("attribute" if sep else "dataset path"))
            declared_shape = entry.get("shape")
            # An attribute is a single value; so is a dataset declared with an
            # empty shape, which is how a 0-d dataset is spelled.
            kind = "scalar" if sep or declared_shape == [] else "array"
            field = self._register(Field(name, name, kind, entry, self._next_order()))
            field.spec = {"at": at, "object": object_path, "attribute": attribute if sep else None}
            if "type" in entry:
                if entry["type"] not in TEXT_TYPES:
                    raise SchemaError(f"{spot}.type: unknown type {entry['type']!r} "
                                      f"(known: {', '.join(sorted(TEXT_TYPES))})")
                field.type = entry["type"]
            else:
                self.assumptions.append(f'field "{name}": type taken from the file [default]')
            if declared_shape:
                field.shape = self._shape(declared_shape, "", field.order, spot)
            self._value_options(field, entry, spot)
            if "{index}" in at:
                self.index_dependent = True

    # -- CSV --------------------------------------------------------------

    def _build_csv(self, block: dict) -> None:
        """Build the flat field list for a CSV schema."""
        _check_keys(block, ["format", "fields"], CSV_INPUT_KEYS, "schema.input")
        header = block.get("header", True)
        if not isinstance(header, bool):
            raise SchemaError("schema.input.header: expected true or false")
        delimiter = block.get("delimiter", ",")
        if not isinstance(delimiter, str) or len(delimiter) != 1:
            raise SchemaError("schema.input.delimiter: expected a single character")
        skip_lines = block.get("skip_lines", 0)
        if not isinstance(skip_lines, int) or isinstance(skip_lines, bool) or skip_lines < 0:
            raise SchemaError("schema.input.skip_lines: expected a non-negative integer")
        self.csv_options = {"header": header, "delimiter": delimiter, "skip_lines": skip_lines}
        for key, shown in (("header", "true"), ("delimiter", '","'), ("skip_lines", "0")):
            if key not in block:
                self.assumptions.append(f"csv {key}={shown} [default]")

        # $nrows is available to CSV expressions and is known before any field.
        nrows = self._register(Field("nrows", "nrows", "pseudo", {}, self._next_order()))
        nrows.type = "u64"

        entries = block["fields"]
        if not isinstance(entries, list) or not entries:
            raise SchemaError("schema.input.fields: expected a non-empty list of columns")
        for position, entry in enumerate(entries):
            spot = f"schema.input.fields[{position}]"
            _check_keys(entry, [], CSV_FIELD_KEYS, spot)
            has_one = "column" in entry
            has_many = "columns" in entry
            if has_one == has_many:
                raise SchemaError(f"{spot}: expected exactly one of 'column' or 'columns'")
            if has_many:
                columns = entry["columns"]
                if not isinstance(columns, list) or len(columns) < 1:
                    raise SchemaError(f"{spot}.columns: expected a non-empty list")
                if "name" not in entry:
                    raise SchemaError(f"{spot}: 'columns' requires an explicit 'name'")
            else:
                columns = [entry["column"]]
            for column in columns:
                if not isinstance(column, (str, int)) or isinstance(column, bool):
                    raise SchemaError(f"{spot}: column {column!r} must be a header name "
                                      f"or a zero-based position")
            if "name" in entry:
                name = _check_name(entry["name"], f"{spot}.name")
            elif isinstance(columns[0], str):
                name = _check_name(columns[0], f"{spot}: default name {columns[0]!r}")
                self.assumptions.append(f'field "{name}": name from column header')
            else:
                raise SchemaError(f"{spot}: a positional column requires an explicit 'name'")
            row = entry.get("row")
            if row is not None and (not isinstance(row, int) or isinstance(row, bool) or row < 0):
                raise SchemaError(f"{spot}.row: expected a non-negative integer")
            kind = "array" if ("shape" in entry or has_many) else "scalar"
            if row is not None and kind == "array":
                raise SchemaError(f"{spot}.row: selects a single cell, so it cannot be "
                                  f"combined with 'shape' or 'columns'")
            field = self._register(Field(name, name, kind, entry, self._next_order()))
            field.spec = {"columns": columns, "row": row, "multi": has_many}
            field.type = entry.get("type", "str")
            if field.type not in TEXT_TYPES:
                raise SchemaError(f"{spot}.type: unknown type {field.type!r} "
                                  f"(known: {', '.join(sorted(TEXT_TYPES))})")
            if "type" not in entry:
                hint = " - declare type: f32 or similar for numeric output"
                self.assumptions.append(f'field "{name}": type str [default]{hint}')
            if "shape" in entry:
                field.shape = self._shape(entry["shape"], "", field.order, spot)
            self._value_options(field, entry, spot)
            if kind == "scalar" and row is None:
                self.index_dependent = True

    # -- output -----------------------------------------------------------

    def leaf_paths(self) -> list[str]:
        """Every emittable field path, in declaration order.

        A repeating structure is emitted whole; its members are addressable too,
        each yielding one value per element, so both appear here.
        """
        return [field.path for field in
                sorted(self.index.values(), key=lambda item: item.order)
                if field.is_value]

    def inside_list(self, path: str) -> str | None:
        """Return the path of the repeating structure containing `path`, if any."""
        parts = path.split(".")[:-1]
        while parts:
            candidate = ".".join(parts)
            field = self.index.get(candidate)
            if field is not None and field.kind == "list":
                return candidate
            parts = parts[:-1]
        return None

    def _build_output(self, block: Any) -> OutputSpec:
        """Build and validate the output half of the schema."""
        if block is None:
            block = {}
            self.assumptions.append("no output block: json to standard output [default]")
        _check_keys(block, [], OUTPUT_KEYS, "schema.output")
        fmt = block.get("format", "json")
        if fmt not in OUTPUT_FORMATS:
            raise SchemaError(f"schema.output.format: expected one of "
                              f"{', '.join(OUTPUT_FORMATS)}, got {fmt!r}")
        path = block.get("path")
        if path is not None and not isinstance(path, str):
            raise SchemaError("schema.output.path: expected a string")
        if path:
            path = self.expand_placeholders(path)

        entries = block.get("fields")
        if entries is None:
            # A repeating structure is emitted whole by default; its members are
            # addressable but would only duplicate what the structure already
            # carries, so they are left out unless asked for by name.
            fields = [(item, item) for item in self.leaf_paths()
                      if self.inside_list(item) is None]
            self.assumptions.append("no output.fields: every field emitted, keyed by full path")
        else:
            if not isinstance(entries, list) or not entries:
                raise SchemaError("schema.output.fields: expected a non-empty list")
            fields = []
            for position, entry in enumerate(entries):
                spot = f"schema.output.fields[{position}]"
                if isinstance(entry, str):
                    fields.append((entry, entry))
                elif isinstance(entry, dict):
                    _check_keys(entry, ["field"], ["as"], spot)
                    target = _check_str(entry["field"], f"{spot}.field")
                    fields.append((target, _check_str(entry.get("as", target), f"{spot}.as")))
                else:
                    raise SchemaError(f"{spot}: expected a field path or a "
                                      f"{{field: ..., as: ...}} mapping")

        seen: set[str] = set()
        for target, out_name in fields:
            if target not in RESERVED_FIELDS and target not in self.index:
                raise SchemaError(f"schema.output.fields: {target!r} is not a field in this "
                                  f"schema (available: {', '.join(self.leaf_paths())})")
            if target not in RESERVED_FIELDS and not self.index[target].is_value:
                raise SchemaError(f"schema.output.fields: {target!r} is a nested structure, "
                                  f"not a value")
            if out_name in seen:
                raise SchemaError(f"schema.output.fields: duplicate output name {out_name!r}")
            seen.add(out_name)

        arrays = [target for target, _ in fields
                  if target not in RESERVED_FIELDS and self.index[target].is_array
                  and self.inside_list(target) is None]
        repeated = [target for target, _ in fields
                    if target not in RESERVED_FIELDS
                    and (self.index[target].kind == "list"
                         or self.inside_list(target) is not None)]
        if fmt == "csv" and (arrays or repeated):
            raise SchemaError(f"schema.output: csv cannot hold multi-valued field(s) "
                              f"{', '.join(arrays + repeated)}; use json")
        if fmt in ARRAY_OUTPUTS and repeated:
            raise SchemaError(f"schema.output: {fmt} output cannot hold field(s) "
                              f"{', '.join(repeated)} from a repeating structure; use json")
        if fmt in ARRAY_OUTPUTS:
            if len(fields) != 1:
                raise SchemaError(f"schema.output: {fmt} output takes exactly one array "
                                  f"field, got {len(fields)}")
            if not arrays:
                raise SchemaError(f"schema.output: {fmt} output requires an array field, "
                                  f"but {fields[0][0]!r} is not one")
        return OutputSpec(fmt, path, fields)


def _check_str(value: Any, where: str) -> str:
    """Verify a value is a non-empty string."""
    if not isinstance(value, str) or not value.strip():
        raise SchemaError(f"{where}: expected a non-empty string")
    return value


def load_schema(path: str) -> Schema:
    """Load and validate one schema file."""
    try:
        with open(path, encoding="utf-8") as handle:
            document = yaml.safe_load(handle)
    except FileNotFoundError as exc:
        raise SchemaError(f"schema file not found: {path}") from exc
    except OSError as exc:
        raise SchemaError(f"cannot read schema file {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise SchemaError(f"invalid YAML in {path}: {exc}") from exc
    if document is None:
        raise SchemaError(f"schema file is empty: {path}")
    if not isinstance(document, dict):
        raise SchemaError(f"{path}: expected a mapping at the top level")
    try:
        schema = Schema(document, path)
    except SchemaError as exc:
        raise SchemaError(f"{path}: {exc}") from exc
    log.debug("loaded schema %r from %s (%d fields)", schema.name, path, len(schema.index))
    return schema
