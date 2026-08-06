"""Test suite for structread.

Two layers. The example matrix reads every file under examples/ and converts it
to every output format it legally supports, so the shipped examples cannot drift
out of date. The targeted tests exercise the expression parser, name resolution,
schema validation, damaged input, and every command-line option.
"""

from __future__ import annotations

import csv as csvmod
import glob
import io
import json
import os
import random
import struct
import sys
import tracemalloc
from typing import Any, ClassVar

import numpy as np
import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import structread as sr

EXAMPLES = os.path.join(ROOT, "examples")
HAVE_H5PY = True
try:
    import h5py
except ImportError:
    HAVE_H5PY = False

needs_h5py = pytest.mark.skipif(not HAVE_H5PY, reason="h5py is not installed")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

class _Stdout:
    """Stand-in for sys.stdout that captures text and bytes separately."""

    def __init__(self):
        self.text = io.StringIO()
        self.buffer = io.BytesIO()

    def write(self, value: str) -> int:
        """Capture text output."""
        return self.text.write(value)

    def flush(self) -> None:
        """Present for interface compatibility."""


def run_cli(argv: list[str]) -> tuple[int, str, bytes]:
    """Invoke the entry point, returning (exit code, text output, byte output)."""
    stream = _Stdout()
    code = sr.main(argv, stdout=stream)
    return code, stream.text.getvalue(), stream.buffer.getvalue()


def example(*parts: str) -> str:
    """Path to a file in the example gallery."""
    return os.path.join(EXAMPLES, *parts)


def write_schema(tmp_path, document: dict, name: str = "s.yaml") -> str:
    """Write a schema dictionary to disk and return its path."""
    path = tmp_path / name
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return str(path)


def load_yaml(path: str) -> dict:
    """Read a YAML document from disk."""
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def minimal(fmt: str = "bin") -> dict:
    """A smallest-possible valid schema, used as an edit base."""
    if fmt == "bin":
        inner = {"format": "bin", "header": [{"name": "x", "type": "u8"}]}
    elif fmt == "csv":
        inner = {"format": "csv", "fields": [{"column": "a"}]}
    else:
        inner = {"format": "hdf5", "fields": [{"at": "/@a"}]}
    return {"name": "t", "input": inner}


# --------------------------------------------------------------------------
# Example matrix
# --------------------------------------------------------------------------

# (schema, input, result count with --index all, an index that exists)
GALLERY = [
    ("bin/simple/device.yaml", "bin/simple/device.bin", 1, "0"),
    ("bin/moderate/capture_meta.yaml", "bin/moderate/capture.bin", 3, "0"),
    ("bin/moderate/capture_frame.yaml", "bin/moderate/capture.bin", 3, "0"),
    ("bin/complex/scan.yaml", "bin/complex/scan.bin", 4, "0"),
    ("bin/nested/catalog.yaml", "bin/nested/catalog.bin", 3, "0"),
    ("csv/simple/readings.yaml", "csv/simple/readings.csv", 3, "0"),
    ("csv/moderate/sensors_rows.yaml", "csv/moderate/sensors.csv", 4, "0"),
    ("csv/moderate/sensors_trace.yaml", "csv/moderate/sensors.csv", 1, "0"),
    ("csv/complex/log.yaml", "csv/complex/log.csv", 4, "0"),
    ("hdf5/simple/instrument.yaml", "hdf5/simple/instrument.h5", 1, "0"),
    ("hdf5/moderate/frames_meta.yaml", "hdf5/moderate/frames.h5", 3, "1"),
    ("hdf5/moderate/frames_image.yaml", "hdf5/moderate/frames.h5", 3, "1"),
    ("hdf5/moderate/frames_thumb.yaml", "hdf5/moderate/frames.h5", 3, "1"),
    ("hdf5/complex/scan.yaml", "hdf5/complex/scan.h5", 3, "1"),
]


def gallery_ids() -> list[str]:
    """Readable parametrisation labels for the gallery."""
    return [entry[0] for entry in GALLERY]


def skip_if_needed(schema_path: str) -> None:
    """Skip an HDF5 gallery case when h5py is absent."""
    if schema_path.startswith("hdf5/") and not HAVE_H5PY:
        pytest.skip("h5py is not installed")


@pytest.mark.parametrize(("schema_path", "input_path", "count", "one_index"), GALLERY,
                         ids=gallery_ids())
class TestGallery:
    """Every shipped example, converted to every output format it supports."""

    def test_validates(self, schema_path, input_path, count, one_index):
        skip_if_needed(schema_path)
        assert run_cli(["-s", example(schema_path), "--validate"])[0] == sr.EXIT_OK

    def test_to_json(self, schema_path, input_path, count, one_index):
        skip_if_needed(schema_path)
        code, out, _ = run_cli(["-s", example(schema_path), example(input_path),
                                "--format", "json", "-q"])
        assert code == sr.EXIT_OK
        records = json.loads(out)
        assert len(records) == count
        assert all(isinstance(record, dict) and record for record in records)

    def test_to_csv(self, schema_path, input_path, count, one_index):
        skip_if_needed(schema_path)
        schema = sr.load_schema(example(schema_path))
        arrays = [target for target, _ in schema.output.fields
                  if target not in sr.RESERVED_FIELDS
                  and (schema.index[target].is_array
                       or schema.index[target].kind == "list"
                       or schema.inside_list(target) is not None)]
        code, out, _ = run_cli(["-s", example(schema_path), example(input_path),
                                "--format", "csv", "-q"])
        if arrays:
            assert code == sr.EXIT_USAGE
            return
        assert code == sr.EXIT_OK
        rows = list(csvmod.reader(io.StringIO(out)))
        assert len(rows) == count + 1
        assert rows[0][0]

    @pytest.mark.parametrize("array_format", ["bin", "npy"])
    def test_to_array_format(self, schema_path, input_path, count, one_index, array_format):
        skip_if_needed(schema_path)
        schema = sr.load_schema(example(schema_path))
        fields = schema.output.fields
        single_array = (len(fields) == 1 and fields[0][0] not in sr.RESERVED_FIELDS
                        and schema.index[fields[0][0]].is_array)
        code, _, payload = run_cli(["-s", example(schema_path), example(input_path),
                                    "--format", array_format, "--index", one_index, "-q"])
        if not single_array:
            assert code == sr.EXIT_USAGE
            return
        assert code == sr.EXIT_OK
        assert len(payload) > 0
        if array_format == "npy":
            assert np.load(io.BytesIO(payload)).size > 0

    def test_single_index_is_a_subset(self, schema_path, input_path, count, one_index):
        skip_if_needed(schema_path)
        code, out, _ = run_cli(["-s", example(schema_path), example(input_path),
                                "--format", "json", "--index", one_index, "-q"])
        assert code == sr.EXIT_OK
        assert len(json.loads(out)) == 1


class TestGalleryValues:
    """Spot checks on what the examples actually produce."""

    def test_bin_simple(self):
        _, out, _ = run_cli(["-s", example("bin/simple/device.yaml"),
                             example("bin/simple/device.bin"), "-q"])
        record = json.loads(out)[0]
        assert record["header.serial"] == "SN-4471"
        assert record["header.active"] is True
        assert record["header.temperature"] == pytest.approx(21.5)

    def test_bin_moderate_csv(self):
        _, out, _ = run_cli(["-s", example("bin/moderate/capture_meta.yaml"),
                             example("bin/moderate/capture.bin"), "-q"])
        lines = out.strip().splitlines()
        assert lines[0] == "stem,index,label,frame_id,width,height,timestamp"
        assert lines[1] == "capture,0,sample-capture,1,4,3,1001.0"

    def test_bin_moderate_bytes_match_source(self):
        _, _, payload = run_cli(["-s", example("bin/moderate/capture_frame.yaml"),
                                 example("bin/moderate/capture.bin"), "--index", "1", "-q"])
        expected = (np.arange(12, dtype=np.int64) + 20).astype("<u2")
        assert payload == expected.tobytes()

    def test_bin_complex_scope_and_endian(self):
        _, out, _ = run_cli(["-s", example("bin/complex/scan.yaml"),
                             example("bin/complex/scan.bin"), "-q"])
        records = json.loads(out)
        assert records[0]["header.limits.large"] == -64000000000
        assert records[0]["header.flags.calibrated"] is True
        assert records[0]["header.flags.inverted"] is False
        # Records are variable length: each cube uses its own rows and cols with
        # the plane count reached through an alias into the header.
        assert [record["record.cube"]["shape"] for record in records] == \
               [[2, 3, 3], [1, 4, 3], [3, 2, 3], [0, 2, 3]]

    def test_csv_simple_defaults_to_text(self):
        _, out, _ = run_cli(["-s", example("csv/simple/readings.yaml"),
                             example("csv/simple/readings.csv"), "-q"])
        assert json.loads(out)[0]["ambient"] == "21.5"

    def test_csv_moderate_empty_cell(self):
        _, out, _ = run_cli(["-s", example("csv/moderate/sensors_rows.yaml"),
                             example("csv/moderate/sensors.csv"), "--format", "json", "-q"])
        records = json.loads(out)
        assert records[3]["ambient"] is None
        assert records[3]["serial"] == "SN-4471"

    def test_csv_moderate_arrays(self):
        _, out, _ = run_cli(["-s", example("csv/moderate/sensors_trace.yaml"),
                             example("csv/moderate/sensors.csv"), "-q"])
        record = json.loads(out)[0]
        assert record["signal"]["shape"] == [4]
        assert record["xyz"]["shape"] == [4, 3]
        rebuilt = np.array(record["xyz"]["data"]).reshape(record["xyz"]["shape"])
        assert rebuilt[1].tolist() == pytest.approx([1.5, 2.5, 3.5])

    def test_csv_complex_dialect(self):
        _, out, _ = run_cli(["-s", example("csv/complex/log.yaml"),
                             example("csv/complex/log.csv"), "-q"])
        records = json.loads(out)
        assert records[0]["note"] == "routine; nothing to report"
        assert "\n" in records[1]["note"]
        assert records[2]["reading"] is None
        assert records[3]["note"] == 'quoted "double" quotes'

    @needs_h5py
    def test_hdf5_natural_order(self):
        _, out, _ = run_cli(["-s", example("hdf5/moderate/frames_meta.yaml"),
                             example("hdf5/moderate/frames.h5"), "-q"])
        lines = out.strip().splitlines()
        assert [line.split(",")[1] for line in lines[1:]] == ["1", "2", "10"]

    @needs_h5py
    def test_hdf5_complex_indexed_name_segment(self):
        _, out, _ = run_cli(["-s", example("hdf5/complex/scan.yaml"),
                             example("hdf5/complex/scan.h5"), "-q"])
        records = json.loads(out)
        assert [record["exposure"] for record in records] == [100, 200, 300]
        assert records[0]["calibration"] == pytest.approx(0.977)
        assert records[0]["image"]["shape"] == [3, 5]

    @needs_h5py
    def test_merge_two_formats(self):
        code, out, _ = run_cli(["-s", example("merge/legacy.yaml"),
                                example("bin/moderate/capture.bin"),
                                "-s", example("merge/modern.yaml"),
                                example("hdf5/moderate/frames.h5"), "-q"])
        assert code == sr.EXIT_OK
        lines = out.strip().splitlines()
        assert lines[0] == "schema,stem,index,frame_id,timestamp"
        assert [line.split(",")[0] for line in lines[1:]] == \
               ["legacy"] * 3 + ["modern"] * 3


# --------------------------------------------------------------------------
# Expressions
# --------------------------------------------------------------------------

class TestExpressions:
    """Arithmetic parsing and evaluation."""

    @staticmethod
    def compile(text):
        """Compile an expression whose references resolve to themselves."""
        return sr.compile_expression(text, lambda name: name, "test")

    @pytest.mark.parametrize(("text", "expected"), [
        ("1", 1),
        ("0x10", 16),
        ("$a", 6),
        ("$a + $b", 10),
        ("$a - $b", 2),
        ("$a * $b", 24),
        ("$a / 3", 2),
        ("$a / 4", 1.5),
        ("2 + 3 * 4", 14),
        ("(2 + 3) * 4", 20),
        ("-$b + 10", 6),
        ("  $a  *  ( $b - 2 ) ", 12),
    ])
    def test_valid(self, text, expected):
        assert self.compile(text).evaluate({"a": 6, "b": 4}) == expected

    def test_literal_int(self):
        assert self.compile(7).evaluate({}) == 7

    @pytest.mark.parametrize("text", ["$a +", "1 2", "((1)", "$a #", "", "   ", "* 3", "$a )"])
    def test_malformed(self, text):
        with pytest.raises(sr.SchemaError):
            self.compile(text)

    @pytest.mark.parametrize("value", [True, [1], None])
    def test_wrong_literal_type(self, value):
        with pytest.raises(sr.SchemaError):
            self.compile(value)

    def test_division_by_zero(self):
        with pytest.raises(sr.SchemaError):
            self.compile("$a / 0").evaluate({"a": 1})

    def test_missing_value(self):
        with pytest.raises(sr.DataError):
            self.compile("$a").evaluate({})

    def test_non_numeric_value(self):
        with pytest.raises(sr.DataError):
            self.compile("$a").evaluate({"a": "text"})

    def test_as_int_rejects_fraction(self):
        with pytest.raises(sr.DataError):
            self.compile("7 / 2").as_int({}, "count")

    def test_as_int_rejects_negative(self):
        with pytest.raises(sr.DataError):
            self.compile("0 - 3").as_int({}, "count")

    def test_as_int_accepts_integral_float(self):
        assert self.compile("5.0").as_int({}, "count") == 5

    def test_repr(self):
        assert "1 + 1" in repr(self.compile("1 + 1"))


# --------------------------------------------------------------------------
# Names, scope and aliases
# --------------------------------------------------------------------------

def scoped_schema(shape_expr: str) -> dict:
    """A nested binary schema whose array shape uses the given expression."""
    return {
        "name": "t",
        "input": {
            "format": "bin",
            "header": [
                {"name": "outer", "type": "u8"},
                {"name": "geo", "fields": [
                    {"name": "inner", "type": "u8"},
                    {"name": "same", "type": "u8"},
                ]},
            ],
            "record": [
                {"name": "same", "type": "u8"},
                {"name": "data", "type": "u8", "shape": [shape_expr]},
            ],
        },
    }


class TestScope:
    """Reference resolution across nested scopes."""

    def test_sibling_and_enclosing(self, tmp_path):
        document = scoped_schema("1")
        document["input"]["header"][1]["fields"].append(
            {"name": "buf", "type": "u8", "shape": ["$inner * $outer"]})
        schema = sr.Schema(document, "t.yaml")
        assert schema.index["header.geo.buf"].shape[0].refs == \
               ["header.geo.inner", "header.outer"]

    def test_full_path_from_another_branch(self, tmp_path):
        schema = sr.Schema(scoped_schema("$header.geo.inner"), "t.yaml")
        assert schema.index["record.data"].shape[0].refs == ["header.geo.inner"]

    def test_short_name_from_another_branch_is_rejected(self):
        with pytest.raises(sr.SchemaError, match="does not name any field"):
            sr.Schema(scoped_schema("$inner"), "t.yaml")

    def test_current_scope_wins_only_when_unambiguous(self):
        # 'same' exists in both header.geo and record; from record it is unique.
        schema = sr.Schema(scoped_schema("$same"), "t.yaml")
        assert schema.index["record.data"].shape[0].refs == ["record.same"]

    def test_ambiguity_is_reported(self):
        document = scoped_schema("1")
        document["input"]["header"][1]["fields"].append(
            {"name": "outer", "type": "u8"})
        document["input"]["header"][1]["fields"].append(
            {"name": "buf", "type": "u8", "shape": ["$outer"]})
        with pytest.raises(sr.SchemaError, match="ambiguous"):
            sr.Schema(document, "t.yaml")

    def test_forward_reference_rejected(self):
        document = minimal()
        document["input"]["header"] = [
            {"name": "data", "type": "u8", "shape": ["$later"]},
            {"name": "later", "type": "u8"},
        ]
        with pytest.raises(sr.SchemaError, match="does not name any field"):
            sr.Schema(document, "t.yaml")

    def test_self_reference_rejected(self):
        document = minimal()
        document["input"]["header"] = [{"name": "data", "type": "u8", "shape": ["$data"]}]
        with pytest.raises(sr.SchemaError, match="read later"):
            sr.Schema(document, "t.yaml")

    def test_reference_to_an_array_rejected(self):
        document = minimal()
        document["input"]["header"] = [
            {"name": "buf", "type": "u8", "shape": [2]},
            {"name": "data", "type": "u8", "shape": ["$buf"]},
        ]
        with pytest.raises(sr.SchemaError, match="not a single value"):
            sr.Schema(document, "t.yaml")


class TestAliases:
    """Alias expansion and its guards."""

    def test_expansion_in_expression(self):
        document = scoped_schema("$deep")
        document["aliases"] = {"deep": "header.geo.inner"}
        schema = sr.Schema(document, "t.yaml")
        assert schema.index["record.data"].shape[0].refs == ["header.geo.inner"]

    def test_expansion_in_path(self):
        document = minimal("hdf5")
        document["aliases"] = {"n": "index"}
        document["input"]["fields"] = [{"at": "/data/f{n}@v", "name": "v"}]
        schema = sr.Schema(document, "t.yaml")
        assert schema.index["v"].spec["at"] == "/data/f{index}@v"
        assert schema.index_dependent

    @pytest.mark.parametrize("aliases", [
        {"index": "something"},
        {"a": "b", "b": "c"},
        {"1bad": "x"},
        {"a": ""},
        {"a": 7},
    ])
    def test_rejections(self, aliases):
        document = minimal()
        document["aliases"] = aliases
        with pytest.raises(sr.SchemaError):
            sr.Schema(document, "t.yaml")

    def test_aliases_must_be_a_mapping(self):
        document = minimal()
        document["aliases"] = ["n"]
        with pytest.raises(sr.SchemaError):
            sr.Schema(document, "t.yaml")

    def test_unresolved_alias_is_reported(self):
        document = scoped_schema("$deep")
        document["aliases"] = {"deep": "header.nowhere"}
        with pytest.raises(sr.SchemaError, match="does not name any field"):
            sr.Schema(document, "t.yaml")


# --------------------------------------------------------------------------
# Schema validation
# --------------------------------------------------------------------------

class TestSchemaValidation:
    """Every documented rejection rule."""

    def test_minimal_is_valid(self):
        schema = sr.Schema(minimal(), "t.yaml")
        assert schema.output.format == "json"
        assert schema.leaf_paths() == ["header.x"]

    @pytest.mark.parametrize("mutate", [
        lambda d: d.pop("name"),
        lambda d: d.pop("input"),
        lambda d: d.update(name=""),
        lambda d: d.update(extra=1),
        lambda d: d.update(input={"format": "xml"}),
        lambda d: d.update(input={"nope": 1}),
        lambda d: d.update(input="text"),
    ])
    def test_top_level(self, mutate):
        document = minimal()
        mutate(document)
        with pytest.raises(sr.SchemaError):
            sr.Schema(document, "t.yaml")

    def test_binary_needs_header_or_record(self):
        with pytest.raises(sr.SchemaError, match="header"):
            sr.Schema({"name": "t", "input": {"format": "bin"}}, "t.yaml")

    def test_bad_endian(self):
        document = minimal()
        document["input"]["endian"] = "middle"
        with pytest.raises(sr.SchemaError):
            sr.Schema(document, "t.yaml")

    @pytest.mark.parametrize("field", [
        {"name": "x"},
        {"type": "u8"},
        {"name": "x", "type": "u128"},
        {"name": "x", "type": "char"},
        {"name": "x", "type": "u8", "bogus": 1},
        {"name": "1x", "type": "u8"},
        {"name": "index", "type": "u8"},
        {"name": "source", "type": "u8"},
        {"pad": -1},
        {"pad": 1, "name": "x"},
        {"name": "x", "fields": []},
        {"name": "x", "fields": [{"name": "y", "type": "u8"}], "type": "u8"},
        "not-a-mapping",
    ])
    def test_binary_field_rejections(self, field):
        document = minimal()
        document["input"]["header"] = [field]
        with pytest.raises(sr.SchemaError):
            sr.Schema(document, "t.yaml")

    def test_duplicate_field_paths(self):
        document = minimal()
        document["input"]["header"] = [{"name": "x", "type": "u8"},
                                       {"name": "x", "type": "u8"}]
        with pytest.raises(sr.SchemaError, match="duplicate"):
            sr.Schema(document, "t.yaml")

    def test_empty_field_list(self):
        document = minimal()
        document["input"]["header"] = []
        with pytest.raises(sr.SchemaError):
            sr.Schema(document, "t.yaml")

    def test_bad_shape(self):
        document = minimal()
        document["input"]["header"] = [{"name": "x", "type": "u8", "shape": "2"}]
        with pytest.raises(sr.SchemaError, match="shape"):
            sr.Schema(document, "t.yaml")

    def test_record_count_without_record(self):
        document = minimal()
        document["input"]["record_count"] = 3
        with pytest.raises(sr.SchemaError, match="record_count"):
            sr.Schema(document, "t.yaml")

    @pytest.mark.parametrize("field", [
        {},
        {"at": "/@a", "bogus": 1},
        {"at": "/@a", "type": "u128"},
        {"at": "/@a", "name": "1bad"},
        {"at": "/data/x-y"},
        {"at": ""},
    ])
    def test_hdf5_rejections(self, field):
        document = minimal("hdf5")
        document["input"]["fields"] = [field]
        with pytest.raises(sr.SchemaError):
            sr.Schema(document, "t.yaml")

    def test_hdf5_default_names(self):
        document = minimal("hdf5")
        document["input"]["fields"] = [{"at": "/@run_id"}, {"at": "/data/frame"}]
        schema = sr.Schema(document, "t.yaml")
        assert schema.leaf_paths() == ["run_id", "frame"]
        assert schema.index["run_id"].kind == "scalar"
        assert schema.index["frame"].kind == "array"

    def test_hdf5_empty_shape_means_scalar(self):
        document = minimal("hdf5")
        document["input"]["fields"] = [{"at": "/calibration", "shape": []}]
        assert sr.Schema(document, "t.yaml").index["calibration"].kind == "scalar"

    @pytest.mark.parametrize("field", [
        {},
        {"column": "a", "columns": ["b"]},
        {"columns": ["a", "b"]},
        {"column": 0},
        {"column": "a", "type": "char"},
        {"column": "a", "row": -1},
        {"column": "a", "bogus": 1},
        {"column": "a b"},
        {"columns": [], "name": "x"},
        {"column": [1], "name": "x"},
    ])
    def test_csv_rejections(self, field):
        document = minimal("csv")
        document["input"]["fields"] = [field]
        with pytest.raises(sr.SchemaError):
            sr.Schema(document, "t.yaml")

    @pytest.mark.parametrize("option", [
        {"header": "yes"},
        {"delimiter": ";;"},
        {"skip_lines": -1},
        {"bogus": 1},
    ])
    def test_csv_option_rejections(self, option):
        document = minimal("csv")
        document["input"].update(option)
        with pytest.raises(sr.SchemaError):
            sr.Schema(document, "t.yaml")

    def test_csv_defaults(self):
        schema = sr.Schema(minimal("csv"), "t.yaml")
        assert schema.csv_options == {"header": True, "delimiter": ",", "skip_lines": 0}
        assert schema.index["a"].type == "str"
        assert schema.index_dependent

    @pytest.mark.parametrize("output", [
        {"format": "xml"},
        {"path": 7},
        {"fields": []},
        {"fields": [7]},
        {"fields": ["nosuchfield"]},
        {"fields": [{"as": "x"}]},
        {"fields": [{"field": "header.x", "bogus": 1}]},
        {"fields": ["header.x", {"field": "header.x", "as": "header.x"}]},
        {"bogus": 1},
    ])
    def test_output_rejections(self, output):
        document = minimal()
        document["output"] = output
        with pytest.raises(sr.SchemaError):
            sr.Schema(document, "t.yaml")

    def test_output_struct_is_not_a_value(self):
        document = scoped_schema("1")
        document["output"] = {"fields": ["header.geo"]}
        with pytest.raises(sr.SchemaError, match="nested structure"):
            sr.Schema(document, "t.yaml")

    def test_csv_output_rejects_arrays(self):
        document = minimal()
        document["input"]["header"].append({"name": "buf", "type": "u8", "shape": [2]})
        document["output"] = {"format": "csv", "fields": ["header.buf"]}
        with pytest.raises(sr.SchemaError, match="csv cannot hold multi-valued"):
            sr.Schema(document, "t.yaml")

    def test_bin_output_needs_one_array(self):
        document = minimal()
        document["output"] = {"format": "bin", "fields": ["header.x"]}
        with pytest.raises(sr.SchemaError, match="requires an array"):
            sr.Schema(document, "t.yaml")

    def test_bin_output_rejects_several_fields(self):
        document = minimal()
        document["input"]["header"].append({"name": "buf", "type": "u8", "shape": [2]})
        document["output"] = {"format": "bin", "fields": ["header.buf", "header.x"]}
        with pytest.raises(sr.SchemaError, match="exactly one"):
            sr.Schema(document, "t.yaml")

    def test_reserved_fields_allowed_in_output(self):
        document = minimal()
        document["output"] = {"format": "csv",
                              "fields": ["source", "stem", "index", "schema", "header.x"]}
        assert len(sr.Schema(document, "t.yaml").output.fields) == 5


class TestSchemaLoading:
    """Reading schema files from disk."""

    def test_missing_file(self):
        with pytest.raises(sr.SchemaError, match="not found"):
            sr.load_schema("/nonexistent/schema.yaml")

    def test_bad_yaml(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("name: [unclosed\n", encoding="utf-8")
        with pytest.raises(sr.SchemaError, match="invalid YAML"):
            sr.load_schema(str(path))

    def test_empty_file(self, tmp_path):
        path = tmp_path / "empty.yaml"
        path.write_text("", encoding="utf-8")
        with pytest.raises(sr.SchemaError, match="empty"):
            sr.load_schema(str(path))

    def test_not_a_mapping(self, tmp_path):
        path = tmp_path / "list.yaml"
        path.write_text("- a\n- b\n", encoding="utf-8")
        with pytest.raises(sr.SchemaError, match="mapping"):
            sr.load_schema(str(path))

    def test_directory(self, tmp_path):
        with pytest.raises(sr.SchemaError):
            sr.load_schema(str(tmp_path))

    def test_error_names_the_file(self, tmp_path):
        path = write_schema(tmp_path, {"name": "t"})
        with pytest.raises(sr.SchemaError, match=os.path.basename(path)):
            sr.load_schema(path)


# --------------------------------------------------------------------------
# Damaged and unusual input
# --------------------------------------------------------------------------

@pytest.fixture()
def capture_schema(tmp_path) -> str:
    """A counted binary schema matching the moderate capture example."""
    document = load_yaml(example("bin/moderate/capture_meta.yaml"))
    document["output"] = {"format": "json"}
    return write_schema(tmp_path, document, "capture.yaml")


class TestDamagedBinary:
    """Binary files that do not match their schema."""

    def test_truncated(self, capture_schema, tmp_path):
        with open(example("bin/moderate/capture.bin"), "rb") as handle:
            payload = handle.read()
        path = tmp_path / "short.bin"
        path.write_bytes(payload[:-10])
        code, _, _ = run_cli(["-s", capture_schema, str(path), "-q"])
        assert code == sr.EXIT_DATA

    def test_empty_file(self, capture_schema, tmp_path):
        path = tmp_path / "empty.bin"
        path.write_bytes(b"")
        code, _, _ = run_cli(["-s", capture_schema, str(path), "-q"])
        assert code == sr.EXIT_DATA

    def test_missing_file(self, capture_schema):
        code, _, _ = run_cli(["-s", capture_schema, "/nope/missing.bin", "-q"])
        assert code == sr.EXIT_DATA

    def test_directory_as_input(self, capture_schema, tmp_path):
        code, _, _ = run_cli(["-s", capture_schema, str(tmp_path), "-q"])
        assert code == sr.EXIT_DATA

    def test_index_past_the_end(self, capture_schema):
        code, _, _ = run_cli(["-s", capture_schema, example("bin/moderate/capture.bin"),
                              "--index", "9", "-q"])
        assert code == sr.EXIT_DATA

    def test_trailing_bytes_warn(self, tmp_path):
        # A schema with no record_count reads to end of file, so extra bytes
        # that do not form a whole record are reported.
        document = load_yaml(example("bin/moderate/capture_meta.yaml"))
        del document["input"]["record_count"]
        document["output"] = {"format": "json"}
        path = write_schema(tmp_path, document, "eof.yaml")
        with open(example("bin/moderate/capture.bin"), "rb") as handle:
            payload = handle.read()
        target = tmp_path / "extra.bin"
        target.write_bytes(payload + b"\x00\x00")
        code, out, _ = run_cli(["-s", path, str(target)])
        assert code == sr.EXIT_OK
        assert len(json.loads(out)) == 3

    def test_zero_length_record_guard(self, tmp_path):
        path = write_schema(tmp_path, {
            "name": "t",
            "input": {"format": "bin", "record": [{"pad": 0}]},
            "output": {"format": "json", "fields": ["index"]},
        })
        payload = tmp_path / "b.bin"
        payload.write_bytes(b"\x00\x00")
        code, _, _ = run_cli(["-s", path, str(payload), "-q"])
        assert code == sr.EXIT_SCHEMA

    def test_shape_from_a_corrupt_length(self, tmp_path):
        path = write_schema(tmp_path, {
            "name": "t",
            "input": {"format": "bin", "header": [
                {"name": "count", "type": "u16"},
                {"name": "data", "type": "u8", "shape": ["$count"]},
            ]},
        })
        payload = tmp_path / "b.bin"
        payload.write_bytes(struct.pack("<H", 9999) + b"\x01\x02")
        code, _, _ = run_cli(["-s", path, str(payload), "-q"])
        assert code == sr.EXIT_DATA


class TestDamagedCsv:
    """CSV files that do not match their schema."""

    def test_unparseable_cell(self, tmp_path):
        path = write_schema(tmp_path, {
            "name": "t",
            "input": {"format": "csv", "fields": [{"column": "a", "type": "f32"}]},
        })
        data = tmp_path / "d.csv"
        data.write_text("a\n1.5\nn/a\n", encoding="utf-8")
        code, _, _ = run_cli(["-s", path, str(data), "-q"])
        assert code == sr.EXIT_DATA

    def test_missing_column(self, tmp_path):
        path = write_schema(tmp_path, {
            "name": "t",
            "input": {"format": "csv", "fields": [{"column": "nope"}]},
        })
        data = tmp_path / "d.csv"
        data.write_text("a\n1\n", encoding="utf-8")
        code, _, _ = run_cli(["-s", path, str(data), "-q"])
        assert code == sr.EXIT_DATA

    def test_named_column_without_header(self, tmp_path):
        path = write_schema(tmp_path, {
            "name": "t",
            "input": {"format": "csv", "header": False,
                      "fields": [{"column": "a", "name": "a"}]},
        })
        data = tmp_path / "d.csv"
        data.write_text("1\n", encoding="utf-8")
        code, _, _ = run_cli(["-s", path, str(data), "-q"])
        assert code == sr.EXIT_DATA

    def test_ragged_row_warns_and_continues(self, tmp_path):
        path = write_schema(tmp_path, {
            "name": "t",
            "input": {"format": "csv", "fields": [{"column": "b"}]},
        })
        data = tmp_path / "d.csv"
        data.write_text("a,b\n1,2\n3\n", encoding="utf-8")
        code, out, _ = run_cli(["-s", path, str(data)])
        assert code == sr.EXIT_OK
        assert [record["b"] for record in json.loads(out)] == ["2", ""]

    def test_no_header_row(self, tmp_path):
        path = write_schema(tmp_path, {
            "name": "t",
            "input": {"format": "csv", "fields": [{"column": "a"}]},
        })
        data = tmp_path / "d.csv"
        data.write_text("", encoding="utf-8")
        code, _, _ = run_cli(["-s", path, str(data), "-q"])
        assert code == sr.EXIT_DATA

    def test_no_data_rows(self, tmp_path):
        path = write_schema(tmp_path, {
            "name": "t",
            "input": {"format": "csv", "fields": [{"column": "a"}]},
        })
        data = tmp_path / "d.csv"
        data.write_text("a\n", encoding="utf-8")
        code, _, _ = run_cli(["-s", path, str(data), "-q"])
        assert code == sr.EXIT_DATA

    def test_empty_cell_in_integer_array(self, tmp_path):
        path = write_schema(tmp_path, {
            "name": "t",
            "input": {"format": "csv", "fields": [
                {"column": "a", "type": "u8", "shape": ["$nrows"]}]},
        })
        data = tmp_path / "d.csv"
        data.write_text("a\n1\n\n2\n", encoding="utf-8")
        code, out, _ = run_cli(["-s", path, str(data), "-q"])
        assert code == sr.EXIT_OK
        assert json.loads(out)[0]["a"]["data"] == [1, 2]

    def test_empty_cell_in_float_array_is_null(self, tmp_path):
        path = write_schema(tmp_path, {
            "name": "t",
            "input": {"format": "csv", "fields": [
                {"column": "a", "type": "f32", "shape": ["$nrows"]},
                {"column": "b", "type": "u8", "shape": ["$nrows"]}]},
        })
        data = tmp_path / "d.csv"
        data.write_text("a,b\n1,1\n,2\n", encoding="utf-8")
        code, out, _ = run_cli(["-s", path, str(data), "-q"])
        assert code == sr.EXIT_OK
        assert json.loads(out)[0]["a"]["data"] == [1.0, None]

    def test_empty_cell_in_integer_column_is_rejected(self, tmp_path):
        path = write_schema(tmp_path, {
            "name": "t",
            "input": {"format": "csv", "fields": [
                {"column": "a", "type": "u8", "shape": ["$nrows"]}]},
        })
        data = tmp_path / "d.csv"
        data.write_text("a,b\n1,1\n,2\n", encoding="utf-8")
        code, _, _ = run_cli(["-s", path, str(data), "-q"])
        assert code == sr.EXIT_DATA

    def test_declared_shape_mismatch(self, tmp_path):
        path = write_schema(tmp_path, {
            "name": "t",
            "input": {"format": "csv", "fields": [
                {"column": "a", "type": "f32", "shape": [99]}]},
        })
        data = tmp_path / "d.csv"
        data.write_text("a\n1\n2\n", encoding="utf-8")
        code, _, _ = run_cli(["-s", path, str(data), "-q"])
        assert code == sr.EXIT_DATA

    def test_not_utf8(self, tmp_path):
        path = write_schema(tmp_path, {
            "name": "t",
            "input": {"format": "csv", "fields": [{"column": "a"}]},
        })
        data = tmp_path / "d.csv"
        data.write_bytes(b"a\n\xff\xfe\n")
        code, _, _ = run_cli(["-s", path, str(data), "-q"])
        assert code == sr.EXIT_DATA


@needs_h5py
class TestDamagedHdf5:
    """HDF5 files that do not match their schema."""

    def test_not_hdf5(self, tmp_path):
        path = tmp_path / "bad.h5"
        path.write_bytes(b"not an hdf5 file")
        code, _, _ = run_cli(["-s", example("hdf5/simple/instrument.yaml"), str(path), "-q"])
        assert code == sr.EXIT_DATA

    def test_missing_attribute(self, tmp_path):
        path = write_schema(tmp_path, {
            "name": "t", "input": {"format": "hdf5", "fields": [{"at": "/@nope"}]},
        })
        code, _, _ = run_cli(["-s", path, example("hdf5/simple/instrument.h5"), "-q"])
        assert code == sr.EXIT_DATA

    def test_missing_object(self, tmp_path):
        path = write_schema(tmp_path, {
            "name": "t", "input": {"format": "hdf5", "fields": [{"at": "/nope@a", "name": "a"}]},
        })
        code, _, _ = run_cli(["-s", path, example("hdf5/simple/instrument.h5"), "-q"])
        assert code == sr.EXIT_DATA

    def test_group_where_a_dataset_was_expected(self, tmp_path):
        path = write_schema(tmp_path, {
            "name": "t", "input": {"format": "hdf5", "fields": [{"at": "/data", "name": "data"}]},
        })
        code, _, _ = run_cli(["-s", path, example("hdf5/moderate/frames.h5"), "-q"])
        assert code == sr.EXIT_DATA

    def test_no_index_matches(self, tmp_path):
        path = write_schema(tmp_path, {
            "name": "t",
            "input": {"format": "hdf5",
                      "fields": [{"at": "/nothing{index}/data", "name": "d"}]},
        })
        code, _, _ = run_cli(["-s", path, example("hdf5/moderate/frames.h5"), "-q"])
        assert code == sr.EXIT_DATA

    def test_declared_type_mismatch(self, tmp_path):
        path = write_schema(tmp_path, {
            "name": "t",
            "input": {"format": "hdf5",
                      "fields": [{"at": "/data/frame1/data", "name": "d", "type": "f32"}]},
        })
        code, _, _ = run_cli(["-s", path, example("hdf5/moderate/frames.h5"), "-q"])
        assert code == sr.EXIT_DATA

    def test_declared_shape_mismatch(self, tmp_path):
        path = write_schema(tmp_path, {
            "name": "t",
            "input": {"format": "hdf5",
                      "fields": [{"at": "/data/frame1/data", "name": "d", "shape": [8, 6]}]},
        })
        code, _, _ = run_cli(["-s", path, example("hdf5/moderate/frames.h5"), "-q"])
        assert code == sr.EXIT_DATA

    def test_string_declared_as_number(self, tmp_path):
        path = write_schema(tmp_path, {
            "name": "t",
            "input": {"format": "hdf5", "fields": [{"at": "/@instrument", "type": "u32"}]},
        })
        code, _, _ = run_cli(["-s", path, example("hdf5/simple/instrument.h5"), "-q"])
        assert code == sr.EXIT_DATA

    def test_number_declared_as_string(self, tmp_path):
        path = write_schema(tmp_path, {
            "name": "t",
            "input": {"format": "hdf5", "fields": [{"at": "/@run_id", "type": "str"}]},
        })
        code, _, _ = run_cli(["-s", path, example("hdf5/simple/instrument.h5"), "-q"])
        assert code == sr.EXIT_DATA

    def test_missing_h5py_reports_dependency(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "h5py", None)
        code, _, _ = run_cli(["-s", example("hdf5/simple/instrument.yaml"),
                              example("hdf5/simple/instrument.h5"), "-q"])
        assert code == sr.EXIT_DEPENDENCY

    def test_other_formats_work_without_h5py(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "h5py", None)
        code, _, _ = run_cli(["-s", example("csv/simple/readings.yaml"),
                              example("csv/simple/readings.csv"), "-q"])
        assert code == sr.EXIT_OK


# --------------------------------------------------------------------------
# Multiple files, multiple schemas
# --------------------------------------------------------------------------

class TestMultipleInputs:
    """Several files in one run, per output format."""

    def test_json_concatenates(self):
        code, out, _ = run_cli(["-s", example("csv/simple/readings.yaml"),
                                example("csv/simple/readings.csv"),
                                example("csv/simple/readings.csv"), "-q"])
        assert code == sr.EXIT_OK
        assert len(json.loads(out)) == 6

    def test_csv_writes_one_header(self):
        code, out, _ = run_cli(["-s", example("csv/moderate/sensors_rows.yaml"),
                                example("csv/moderate/sensors.csv"),
                                example("csv/moderate/sensors.csv"), "-q"])
        assert code == sr.EXIT_OK
        lines = out.strip().splitlines()
        assert len(lines) == 9
        assert lines.count(lines[0]) == 1

    def test_bin_to_stdout_concatenates(self):
        _, _, payload = run_cli(["-s", example("bin/moderate/capture_frame.yaml"),
                                 example("bin/moderate/capture.bin"), "-q"])
        assert len(payload) == 3 * 24

    def test_bin_to_files_needs_a_placeholder(self, tmp_path):
        target = tmp_path / "all.bin"
        code, _, _ = run_cli(["-s", example("bin/moderate/capture_frame.yaml"),
                              example("bin/moderate/capture.bin"), "-o", str(target), "-q"])
        assert code == sr.EXIT_USAGE

    def test_bin_to_templated_files(self, tmp_path):
        target = str(tmp_path / "f_{stem}_{index}.bin")
        code, _, _ = run_cli(["-s", example("bin/moderate/capture_frame.yaml"),
                              example("bin/moderate/capture.bin"), "-o", target, "-q"])
        assert code == sr.EXIT_OK
        produced = sorted(os.listdir(tmp_path))
        assert produced == ["f_capture_0.bin", "f_capture_1.bin", "f_capture_2.bin"]
        with open(tmp_path / "f_capture_2.bin", "rb") as handle:
            assert len(handle.read()) == 24

    def test_single_index_needs_no_placeholder(self, tmp_path):
        target = tmp_path / "one.bin"
        code, _, _ = run_cli(["-s", example("bin/moderate/capture_frame.yaml"),
                              example("bin/moderate/capture.bin"),
                              "--index", "0", "-o", str(target), "-q"])
        assert code == sr.EXIT_OK
        assert target.stat().st_size == 24


class TestMultipleSchemas:
    """Repeated -s groups and the agreement rules."""

    def test_group_needs_files(self, tmp_path):
        path = write_schema(tmp_path, minimal())
        code, _, _ = run_cli(["-s", path, "-q"])
        assert code == sr.EXIT_USAGE

    def test_validate_allows_an_empty_group(self, tmp_path):
        path = write_schema(tmp_path, minimal())
        assert run_cli(["-s", path, "--validate", "-q"])[0] == sr.EXIT_OK

    def test_format_disagreement(self, tmp_path):
        first = write_schema(tmp_path, {**minimal(), "output": {"format": "json"}}, "a.yaml")
        second = write_schema(tmp_path, {**minimal(), "output": {"format": "csv"}}, "b.yaml")
        code, _, _ = run_cli(["-s", first, "x", "-s", second, "y", "--validate", "-q"])
        assert code == sr.EXIT_SCHEMA

    def test_field_disagreement(self, tmp_path):
        base = minimal()
        base["input"]["header"].append({"name": "y", "type": "u8"})
        first = write_schema(tmp_path, {**base, "output": {"fields": ["header.x"]}}, "a.yaml")
        second = write_schema(tmp_path, {**base, "output": {"fields": ["header.y"]}}, "b.yaml")
        code, _, _ = run_cli(["-s", first, "x", "-s", second, "y", "--validate", "-q"])
        assert code == sr.EXIT_SCHEMA

    def test_renaming_makes_schemas_agree(self, tmp_path):
        base = minimal()
        base["input"]["header"].append({"name": "y", "type": "u8"})
        first = write_schema(tmp_path, {
            **base, "output": {"fields": [{"field": "header.x", "as": "v"}]}}, "a.yaml")
        second = write_schema(tmp_path, {
            **base, "output": {"fields": [{"field": "header.y", "as": "v"}]}}, "b.yaml")
        code, _, _ = run_cli(["-s", first, "x", "-s", second, "y", "--validate", "-q"])
        assert code == sr.EXIT_OK

    def test_conflicting_paths(self, tmp_path):
        first = write_schema(tmp_path, {
            **minimal(), "output": {"format": "csv", "path": "a.csv", "fields": ["header.x"]}},
            "a.yaml")
        second = write_schema(tmp_path, {
            **minimal(), "output": {"format": "csv", "path": "b.csv", "fields": ["header.x"]}},
            "b.yaml")
        data = tmp_path / "d.bin"
        data.write_bytes(b"\x01")
        code, _, _ = run_cli(["-s", first, str(data), "-s", second, str(data), "-q"])
        assert code == sr.EXIT_SCHEMA

    def test_format_override_bridges_a_disagreement(self, tmp_path):
        data = tmp_path / "d.bin"
        data.write_bytes(b"\x01")
        first = write_schema(tmp_path, {
            **minimal(), "output": {"format": "json", "fields": ["header.x"]}}, "a.yaml")
        second = write_schema(tmp_path, {
            **minimal(), "output": {"format": "csv", "fields": ["header.x"]}}, "b.yaml")
        code, out, _ = run_cli(["-s", first, str(data), "-s", second, str(data),
                                "--format", "csv", "-q"])
        assert code == sr.EXIT_OK
        assert out.strip().splitlines() == ["header.x", "1", "1"]


class TestSkipErrors:
    """Batch behaviour when one file fails."""

    def test_aborts_by_default(self, tmp_path):
        bad = tmp_path / "bad.csv"
        bad.write_text("a\nn/a\n", encoding="utf-8")
        path = write_schema(tmp_path, {
            "name": "t",
            "input": {"format": "csv", "fields": [{"column": "a", "type": "f32"}]},
        })
        code, _, _ = run_cli(["-s", path, example("csv/simple/readings.csv"), str(bad), "-q"])
        assert code == sr.EXIT_DATA

    def test_continues_when_asked(self, tmp_path):
        good = tmp_path / "good.csv"
        good.write_text("a\n1.5\n", encoding="utf-8")
        bad = tmp_path / "bad.csv"
        bad.write_text("a\nn/a\n", encoding="utf-8")
        path = write_schema(tmp_path, {
            "name": "t",
            "input": {"format": "csv", "fields": [{"column": "a", "type": "f32"}]},
        })
        code, out, _ = run_cli(["-s", path, str(good), str(bad), str(good), "--skip-errors", "-q"])
        assert code == sr.EXIT_DATA
        assert len(json.loads(out)) == 2


# --------------------------------------------------------------------------
# Command-line interface
# --------------------------------------------------------------------------

class TestCli:
    """Options, output destinations and exit statuses."""

    def test_validate_only(self):
        code, _, _ = run_cli(["-s", example("bin/complex/scan.yaml"), "--validate", "-q"])
        assert code == sr.EXIT_OK

    def test_validate_reports_a_bad_schema(self, tmp_path):
        path = write_schema(tmp_path, {"name": "t"})
        code, _, _ = run_cli(["-s", path, "--validate", "-q"])
        assert code == sr.EXIT_SCHEMA

    @pytest.mark.parametrize("index", ["-1", "one", "1.5", ""])
    def test_bad_index(self, index):
        code, _, _ = run_cli(["-s", example("csv/simple/readings.yaml"),
                              example("csv/simple/readings.csv"), "--index", index, "-q"])
        assert code == sr.EXIT_USAGE

    def test_bad_indent(self):
        code, _, _ = run_cli(["-s", example("csv/simple/readings.yaml"),
                              example("csv/simple/readings.csv"), "--indent", "-1", "-q"])
        assert code == sr.EXIT_USAGE

    def test_compact_indent(self):
        _, out, _ = run_cli(["-s", example("csv/simple/readings.yaml"),
                             example("csv/simple/readings.csv"), "--indent", "0", "-q"])
        assert "\n" not in out.strip()

    def test_output_file(self, tmp_path):
        target = tmp_path / "out.json"
        code, out, _ = run_cli(["-s", example("csv/simple/readings.yaml"),
                                example("csv/simple/readings.csv"), "-o", str(target), "-q"])
        assert code == sr.EXIT_OK
        assert out == ""
        assert len(json.loads(target.read_text(encoding="utf-8"))) == 3

    def test_dash_means_stdout(self, tmp_path):
        document = load_yaml(example("csv/moderate/sensors_rows.yaml"))
        document["output"]["path"] = str(tmp_path / "unused.csv")
        path = write_schema(tmp_path, document, "s.yaml")
        code, out, _ = run_cli(["-s", path, example("csv/moderate/sensors.csv"), "-o", "-", "-q"])
        assert code == sr.EXIT_OK
        assert out.startswith("stem,index")
        assert not (tmp_path / "unused.csv").exists()

    def test_placeholder_rejected_for_text_output(self, tmp_path):
        code, _, _ = run_cli(["-s", example("csv/simple/readings.yaml"),
                              example("csv/simple/readings.csv"),
                              "-o", str(tmp_path / "x_{index}.json"), "-q"])
        assert code == sr.EXIT_USAGE

    def test_unwritable_output(self, tmp_path):
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory", encoding="utf-8")
        code, _, _ = run_cli(["-s", example("csv/simple/readings.yaml"),
                              example("csv/simple/readings.csv"),
                              "-o", str(blocker / "out.json"), "-q"])
        assert code == sr.EXIT_DATA

    def test_unwritable_bin_output(self, tmp_path):
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory", encoding="utf-8")
        code, _, _ = run_cli(["-s", example("bin/moderate/capture_frame.yaml"),
                              example("bin/moderate/capture.bin"), "--index", "0",
                              "-o", str(blocker / "f.bin"), "-q"])
        assert code == sr.EXIT_DATA

    def test_format_bin_needs_one_field(self):
        code, _, _ = run_cli(["-s", example("csv/simple/readings.yaml"),
                              example("csv/simple/readings.csv"), "--format", "bin", "-q"])
        assert code == sr.EXIT_USAGE

    def test_format_bin_needs_an_array(self, tmp_path):
        document = minimal()
        document["output"] = {"fields": ["header.x"]}
        path = write_schema(tmp_path, document)
        data = tmp_path / "d.bin"
        data.write_bytes(b"\x01")
        code, _, _ = run_cli(["-s", path, str(data), "--format", "bin", "-q"])
        assert code == sr.EXIT_DATA

    def test_version(self):
        with pytest.raises(SystemExit) as excinfo:
            run_cli(["--version"])
        assert excinfo.value.code == 0

    def test_schema_argument_required(self):
        with pytest.raises(SystemExit) as excinfo:
            run_cli(["input.bin"])
        assert excinfo.value.code == 2


class TestLogging:
    """Verbosity, the startup summary, and stream separation."""

    def test_summary_lists_defaults(self, capsys):
        run_cli(["-s", example("csv/simple/readings.yaml"),
                 example("csv/simple/readings.csv")])
        err = capsys.readouterr().err
        assert f"structread {sr.__version__}" in err
        assert "index     all [default]" in err
        assert "type str [default]" in err
        assert "csv delimiter" in err

    def test_summary_names_unread_fields(self, capsys):
        run_cli(["-s", example("bin/moderate/capture_meta.yaml"),
                 example("bin/moderate/capture.bin")])
        assert "will not be read: header.magic" in capsys.readouterr().err

    def test_quiet_silences_info(self, capsys):
        run_cli(["-s", example("csv/simple/readings.yaml"),
                 example("csv/simple/readings.csv"), "-q"])
        assert capsys.readouterr().err == ""

    def test_verbose_adds_debug(self, capsys):
        run_cli(["-s", example("bin/moderate/capture_meta.yaml"),
                 example("bin/moderate/capture.bin"), "-v"])
        assert "DEBUG" in capsys.readouterr().err

    def test_double_verbose_traces_fields(self, capsys):
        run_cli(["-s", example("bin/moderate/capture_meta.yaml"),
                 example("bin/moderate/capture.bin"), "-vv"])
        err = capsys.readouterr().err
        assert "TRACE" in err
        assert "record.data skipped" in err

    def test_log_file_records_debug_while_quiet(self, tmp_path, capsys):
        target = tmp_path / "run.log"
        run_cli(["-s", example("bin/moderate/capture_meta.yaml"),
                 example("bin/moderate/capture.bin"), "-q", "--log-file", str(target)])
        assert capsys.readouterr().err == ""
        contents = target.read_text(encoding="utf-8")
        assert "DEBUG" in contents
        assert "TRACE" in contents

    def test_unwritable_log_file(self, tmp_path):
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory", encoding="utf-8")
        code, _, _ = run_cli(["-s", example("csv/simple/readings.yaml"),
                              example("csv/simple/readings.csv"),
                              "--log-file", str(blocker / "x.log")])
        assert code == sr.EXIT_USAGE

    def test_bin_output_keeps_stdout_clean(self, capsys):
        _, text, payload = run_cli(["-s", example("bin/moderate/capture_frame.yaml"),
                                    example("bin/moderate/capture.bin"), "--index", "0"])
        assert text == ""
        assert len(payload) == 24
        assert "INFO" in capsys.readouterr().err


# --------------------------------------------------------------------------
# Scale
# --------------------------------------------------------------------------

class TestScale:
    """Behaviour on input far larger than the examples."""

    @staticmethod
    def build(path, records: int, side: int) -> None:
        """Write a capture file with `records` frames of side x side u16 pixels."""
        with open(path, "wb") as handle:
            handle.write(struct.pack("<IHH16s", 0x53545244, 1, records, b"large"))
            pixels = np.zeros(side * side, dtype="<u2").tobytes()
            for number in range(records):
                handle.write(struct.pack("<QHH2xd", number, side, side, float(number)))
                handle.write(pixels)

    def test_scalars_only_skips_array_data(self, tmp_path):
        # 64 frames of 512x512 u16 is 32 MB of pixels; reading only the scalars
        # must not materialise any of it.
        source = tmp_path / "large.bin"
        self.build(source, 64, 512)
        assert source.stat().st_size > 30_000_000
        document = load_yaml(example("bin/moderate/capture_meta.yaml"))
        path = write_schema(tmp_path, document, "meta.yaml")
        code, _, _ = run_cli(["-s", path, str(source), "-o", str(tmp_path / "o.csv"), "-q"])
        assert code == sr.EXIT_OK
        assert len((tmp_path / "o.csv").read_text(encoding="utf-8").splitlines()) == 65

    def test_random_access_to_one_record(self, tmp_path):
        source = tmp_path / "large.bin"
        self.build(source, 64, 512)
        document = load_yaml(example("bin/moderate/capture_frame.yaml"))
        path = write_schema(tmp_path, document, "frame.yaml")
        target = tmp_path / "one.bin"
        code, _, _ = run_cli(["-s", path, str(source), "--index", "63",
                              "-o", str(target), "-q"])
        assert code == sr.EXIT_OK
        assert target.stat().st_size == 512 * 512 * 2


# --------------------------------------------------------------------------
# Units and remaining branches
# --------------------------------------------------------------------------

class TestUnits:
    """Small helpers exercised directly."""

    def test_numpy_dtype(self):
        assert sr.numpy_dtype("char") == np.dtype("uint8")
        assert sr.numpy_dtype("u8", ">").itemsize == 1
        assert sr.numpy_dtype("u16", ">").byteorder in (">", "=")

    def test_type_size(self):
        assert sr.type_size("char") == 1
        assert sr.type_size("f64") == 8

    def test_jsonable(self):
        assert sr.jsonable(np.uint16(5)) == 5
        assert sr.jsonable(b"hi") == "hi"
        assert sr.jsonable(np.array([1, 2])) == [1, 2]
        assert sr.jsonable((1, np.int8(2))) == [1, 2]
        assert sr.jsonable(float("nan")) is None
        assert sr.jsonable("plain") == "plain"

    def test_field_repr(self):
        field = sr.Field("a.b", "b", "scalar", {}, 1)
        assert "a.b" in repr(field)

    def test_output_signature(self):
        schema = sr.Schema(minimal(), "t.yaml")
        assert schema.output.signature() == ("json", (("header.x", "header.x"),))

    def test_unclosed_parenthesis(self):
        with pytest.raises(sr.SchemaError, match=r"expected '\)'"):
            sr.compile_expression("(1 2)", lambda name: name, "test")

    def test_parse_index(self):
        assert sr.parse_index("all") is None
        assert sr.parse_index("7") == 7

    @pytest.mark.parametrize("fmt", ["hdf5", "csv"])
    def test_fields_must_be_a_list(self, fmt):
        document = minimal(fmt)
        document["input"]["fields"] = {}
        with pytest.raises(sr.SchemaError, match="non-empty list"):
            sr.Schema(document, "t.yaml")

    @pytest.mark.parametrize("fmt", ["hdf5", "csv"])
    def test_field_entry_must_be_a_mapping(self, fmt):
        document = minimal(fmt)
        document["input"]["fields"] = ["oops"]
        with pytest.raises(sr.SchemaError, match="mapping"):
            sr.Schema(document, "t.yaml")

    def test_alias_with_a_dotted_tail(self):
        document = scoped_schema("$g.inner")
        document["aliases"] = {"g": "header.geo"}
        schema = sr.Schema(document, "t.yaml")
        assert schema.index["record.data"].shape[0].refs == ["header.geo.inner"]

    def test_csv_boolean_column(self, tmp_path):
        path = write_schema(tmp_path, {
            "name": "t",
            "input": {"format": "csv", "fields": [{"column": "flag", "type": "bool"}]},
        })
        data = tmp_path / "d.csv"
        data.write_text("flag\ntrue\nno\n1\n", encoding="utf-8")
        code, out, _ = run_cli(["-s", path, str(data), "-q"])
        assert code == sr.EXIT_OK
        assert [record["flag"] for record in json.loads(out)] == [True, False, True]

    def test_csv_boolean_rejects_other_text(self, tmp_path):
        path = write_schema(tmp_path, {
            "name": "t",
            "input": {"format": "csv", "fields": [{"column": "flag", "type": "bool"}]},
        })
        data = tmp_path / "d.csv"
        data.write_text("flag\nmaybe\n", encoding="utf-8")
        assert run_cli(["-s", path, str(data), "-q"])[0] == sr.EXIT_DATA

    def test_csv_text_array(self, tmp_path):
        path = write_schema(tmp_path, {
            "name": "t",
            "input": {"format": "csv", "fields": [
                {"columns": ["a", "b"], "name": "pair"}]},
        })
        data = tmp_path / "d.csv"
        data.write_text("a,b\nx,y\np,q\n", encoding="utf-8")
        code, out, _ = run_cli(["-s", path, str(data), "-q"])
        assert code == sr.EXIT_OK
        pair = json.loads(out)[0]["pair"]
        assert pair["dtype"] == "str"
        assert pair["data"] == ["x", "y", "p", "q"]

    def test_csv_pinned_row_past_the_end(self, tmp_path):
        path = write_schema(tmp_path, {
            "name": "t",
            "input": {"format": "csv", "fields": [{"column": "a", "row": 9}]},
        })
        data = tmp_path / "d.csv"
        data.write_text("a\n1\n", encoding="utf-8")
        assert run_cli(["-s", path, str(data), "-q"])[0] == sr.EXIT_DATA

    def test_csv_directory_as_input(self, tmp_path):
        path = write_schema(tmp_path, minimal("csv"))
        assert run_cli(["-s", path, str(tmp_path), "-q"])[0] == sr.EXIT_DATA

    @needs_h5py
    def test_hdf5_unlisted_fields_are_not_read(self, tmp_path, capsys):
        document = load_yaml(example("hdf5/complex/scan.yaml"))
        document["output"] = {"format": "json", "fields": ["exposure"]}
        path = write_schema(tmp_path, document, "subset.yaml")
        code, out, _ = run_cli(["-s", path, example("hdf5/complex/scan.h5"), "-vv"])
        assert code == sr.EXIT_OK
        assert list(json.loads(out)[0]) == ["exposure"]
        assert "image skipped (not in output)" in capsys.readouterr().err

    @needs_h5py
    def test_array_attribute_rejected_by_csv(self, tmp_path):
        source = tmp_path / "arr.h5"
        with h5py.File(source, "w") as handle:
            handle.attrs["corners"] = np.array([1, 2, 3, 4], dtype=np.uint16)
        path = write_schema(tmp_path, {
            "name": "t",
            "input": {"format": "hdf5", "fields": [{"at": "/@corners"}]},
            "output": {"format": "csv", "fields": ["corners"]},
        })
        code, _, _ = run_cli(["-s", path, str(source), "-q"])
        assert code == sr.EXIT_DATA

    def test_csv_text_array_with_empty_cells(self, tmp_path):
        path = write_schema(tmp_path, {
            "name": "t",
            "input": {"format": "csv", "fields": [
                {"column": "a", "name": "letters", "shape": ["$nrows"]}]},
        })
        data = tmp_path / "d.csv"
        data.write_text("a,b\nx,1\n,2\n", encoding="utf-8")
        code, out, _ = run_cli(["-s", path, str(data), "-q"])
        assert code == sr.EXIT_OK
        assert json.loads(out)[0]["letters"]["data"] == ["x", ""]

    @pytest.mark.parametrize(("error", "expected"), [
        (BrokenPipeError, sr.EXIT_OK),
        (KeyboardInterrupt, sr.EXIT_DATA),
        (OSError("device gone"), sr.EXIT_DATA),
    ])
    def test_interruptions_map_to_exit_codes(self, monkeypatch, error, expected):
        def explode(*_args, **_kwargs):
            """Stand in for a run that fails partway through."""
            raise error

        monkeypatch.setattr(sr, "execute", explode)
        code, _, _ = run_cli(["-s", example("csv/simple/readings.yaml"),
                              example("csv/simple/readings.csv"), "-q"])
        assert code == expected


# --------------------------------------------------------------------------
# Schema constructs that the gallery accepts but does not assert
# --------------------------------------------------------------------------

def pack_bin(path, payload: bytes) -> str:
    """Write a binary fixture and return its path."""
    path.write_bytes(payload)
    return str(path)


class TestBinaryConstructs:
    """Binary schema features exercised end to end."""

    def test_native_endian(self, tmp_path):
        path = write_schema(tmp_path, {
            "name": "t",
            "input": {"format": "bin", "endian": "native",
                      "header": [{"name": "v", "type": "u16"}]},
        })
        source = pack_bin(tmp_path / "n.bin", np.array([258], dtype=np.uint16).tobytes())
        code, out, _ = run_cli(["-s", path, source, "-q"])
        assert code == sr.EXIT_OK
        assert json.loads(out)[0]["header.v"] == 258

    @pytest.mark.parametrize("endian", ["little", "big"])
    def test_declared_endian(self, tmp_path, endian):
        path = write_schema(tmp_path, {
            "name": "t",
            "input": {"format": "bin", "endian": endian,
                      "header": [{"name": "v", "type": "u16"}]},
        }, f"{endian}.yaml")
        source = pack_bin(tmp_path / f"{endian}.bin", struct.pack(">H", 1))
        _, out, _ = run_cli(["-s", path, source, "-q"])
        assert json.loads(out)[0]["header.v"] == (1 if endian == "big" else 256)

    def test_record_count_as_a_literal(self, tmp_path):
        path = write_schema(tmp_path, {
            "name": "t",
            "input": {"format": "bin", "record_count": 2,
                      "record": [{"name": "v", "type": "u8"}]},
        })
        source = pack_bin(tmp_path / "c.bin", bytes([7, 8, 9]))
        code, out, _ = run_cli(["-s", path, source, "-q"])
        assert code == sr.EXIT_OK
        assert [record["record.v"] for record in json.loads(out)] == [7, 8]

    def test_record_without_a_header(self, tmp_path):
        path = write_schema(tmp_path, {
            "name": "t",
            "input": {"format": "bin", "record": [{"name": "v", "type": "u16"}]},
            "output": {"format": "csv", "fields": ["index", "record.v"]},
        })
        source = pack_bin(tmp_path / "r.bin", struct.pack("<HH", 1, 2))
        code, out, _ = run_cli(["-s", path, source, "-q"])
        assert code == sr.EXIT_OK
        assert out.strip().splitlines() == ["index,record.v", "0,1", "1,2"]

    def test_trailing_pad_in_a_record(self, tmp_path):
        path = write_schema(tmp_path, {
            "name": "t",
            "input": {"format": "bin",
                      "record": [{"name": "v", "type": "u8"}, {"pad": 1}]},
        })
        source = pack_bin(tmp_path / "p.bin", bytes([1, 0, 2, 0]))
        code, out, _ = run_cli(["-s", path, source, "-q"])
        assert code == sr.EXIT_OK
        assert [record["record.v"] for record in json.loads(out)] == [1, 2]

    def test_char_filling_its_shape(self, tmp_path):
        path = write_schema(tmp_path, {
            "name": "t",
            "input": {"format": "bin",
                      "header": [{"name": "tag", "type": "char", "shape": [4]}]},
        })
        source = pack_bin(tmp_path / "s.bin", b"ABCD")
        _, out, _ = run_cli(["-s", path, source, "-q"])
        assert json.loads(out)[0]["header.tag"] == "ABCD"

    def test_non_finite_float(self, tmp_path):
        path = write_schema(tmp_path, {
            "name": "t",
            "input": {"format": "bin", "header": [
                {"name": "bad", "type": "f32"}, {"name": "good", "type": "f32"}]},
            "output": {"format": "csv", "fields": ["header.bad", "header.good"]},
        })
        source = pack_bin(tmp_path / "n.bin", struct.pack("<ff", float("nan"), 1.5))
        code, out, _ = run_cli(["-s", path, source, "-q"])
        assert code == sr.EXIT_OK
        assert out.strip().splitlines()[1] == ",1.5"
        _, out, _ = run_cli(["-s", path, source, "--format", "json", "-q"])
        assert json.loads(out)[0]["header.bad"] is None

    def test_signed_widths_read_back(self):
        _, out, _ = run_cli(["-s", example("bin/complex/scan.yaml"),
                             example("bin/complex/scan.bin"), "--index", "0", "-q"])
        record = json.loads(out)[0]
        assert record["header.limits.smallest"] == -8
        assert record["header.limits.small"] == -1600
        assert record["header.limits.medium"] == -320000
        assert record["header.limits.large"] == -64000000000

    def test_third_level_of_nesting(self):
        _, out, _ = run_cli(["-s", example("bin/complex/scan.yaml"),
                             example("bin/complex/scan.bin"), "--index", "0", "-q"])
        record = json.loads(out)[0]
        assert record["header.geometry.origin.x"] == -12
        assert record["header.geometry.origin.y"] == 34

    def test_arithmetic_and_mixed_shapes(self):
        _, out, _ = run_cli(["-s", example("bin/complex/scan.yaml"),
                             example("bin/complex/scan.bin"), "-q"])
        records = json.loads(out)
        # edges is shape [$rows + $cols]; pair is shape [2, $cols].
        assert [record["record.edges"]["shape"] for record in records] == \
               [[5], [5], [5], [2]]
        assert [record["record.pair"]["shape"] for record in records] == \
               [[2, 3], [2, 4], [2, 2], [2, 2]]

    def test_zero_length_dimension(self):
        _, out, _ = run_cli(["-s", example("bin/complex/scan.yaml"),
                             example("bin/complex/scan.bin"), "--index", "3", "-q"])
        cube = json.loads(out)[0]["record.cube"]
        assert cube["shape"] == [0, 2, 3]
        assert cube["data"] == []


class TestCsvConstructs:
    """CSV schema features exercised end to end."""

    def test_skip_lines_with_a_header(self, tmp_path):
        path = write_schema(tmp_path, {
            "name": "t",
            "input": {"format": "csv", "skip_lines": 2,
                      "fields": [{"column": "a", "type": "i32"}]},
        })
        data = tmp_path / "d.csv"
        data.write_text("# note\n# note\na,b\n-5,x\n", encoding="utf-8")
        code, out, _ = run_cli(["-s", path, str(data), "-q"])
        assert code == sr.EXIT_OK
        assert json.loads(out)[0]["a"] == -5

    def test_tab_delimiter(self, tmp_path):
        path = write_schema(tmp_path, {
            "name": "t",
            "input": {"format": "csv", "delimiter": "\t",
                      "fields": [{"column": "b", "type": "u16"}]},
        })
        data = tmp_path / "d.tsv"
        data.write_text("a\tb\n1\t2\n", encoding="utf-8")
        code, out, _ = run_cli(["-s", path, str(data), "-q"])
        assert code == sr.EXIT_OK
        assert json.loads(out)[0]["b"] == 2

    def test_single_column_group(self, tmp_path):
        path = write_schema(tmp_path, {
            "name": "t",
            "input": {"format": "csv",
                      "fields": [{"columns": ["a"], "name": "only", "type": "u8"}]},
        })
        data = tmp_path / "d.csv"
        data.write_text("a\n1\n2\n", encoding="utf-8")
        code, out, _ = run_cli(["-s", path, str(data), "-q"])
        assert code == sr.EXIT_OK
        assert json.loads(out)[0]["only"]["shape"] == [2, 1]

    def test_index_past_the_last_row(self):
        code, _, _ = run_cli(["-s", example("csv/simple/readings.yaml"),
                              example("csv/simple/readings.csv"), "--index", "99", "-q"])
        assert code == sr.EXIT_DATA

    def test_integer_scalar_value(self, tmp_path):
        path = write_schema(tmp_path, {
            "name": "t",
            "input": {"format": "csv", "fields": [
                {"column": "a", "type": "i16"}, {"column": "b", "type": "u32"}]},
        })
        data = tmp_path / "d.csv"
        data.write_text("a,b\n-300,70000\n", encoding="utf-8")
        _, out, _ = run_cli(["-s", path, str(data), "-q"])
        assert json.loads(out)[0] == {"a": -300, "b": 70000}

    @pytest.mark.parametrize("extra", [{"shape": ["$nrows"]}, {"columns": ["a", "b"]}])
    def test_row_cannot_combine_with_an_array(self, extra):
        field = {"name": "x", "type": "f32", "row": 0}
        field.update(extra)
        if "columns" in field:
            field.pop("column", None)
        else:
            field["column"] = "a"
        document = minimal("csv")
        document["input"]["fields"] = [field]
        with pytest.raises(sr.SchemaError, match="single cell"):
            sr.Schema(document, "t.yaml")


@needs_h5py
class TestHdf5Constructs:
    """HDF5 schema features exercised end to end."""

    @pytest.fixture()
    def mixed(self, tmp_path) -> str:
        """A file with a text dataset, an untyped numeric dataset and attributes."""
        path = tmp_path / "mixed.h5"
        with h5py.File(path, "w") as handle:
            handle.create_dataset("label", data="calibration-run")
            handle.create_dataset("counts", data=np.arange(6, dtype=np.int32).reshape(2, 3))
        return str(path)

    def test_text_dataset(self, tmp_path, mixed):
        path = write_schema(tmp_path, {
            "name": "t",
            "input": {"format": "hdf5",
                      "fields": [{"at": "/label", "type": "str", "shape": []}]},
        })
        code, out, _ = run_cli(["-s", path, mixed, "-q"])
        assert code == sr.EXIT_OK
        assert json.loads(out)[0]["label"] == "calibration-run"

    def test_untyped_dataset_uses_the_file_type(self, tmp_path, mixed):
        path = write_schema(tmp_path, {
            "name": "t", "input": {"format": "hdf5", "fields": [{"at": "/counts"}]},
        })
        code, out, _ = run_cli(["-s", path, mixed, "-q"])
        assert code == sr.EXIT_OK
        counts = json.loads(out)[0]["counts"]
        assert counts["dtype"] == "int32"
        assert counts["shape"] == [2, 3]

    def test_default_name_from_a_dataset_path(self, tmp_path, mixed):
        path = write_schema(tmp_path, {
            "name": "t", "input": {"format": "hdf5", "fields": [{"at": "/counts"}]},
        })
        _, out, _ = run_cli(["-s", path, mixed, "-q"])
        assert list(json.loads(out)[0]) == ["counts"]

    def test_index_naming_a_missing_frame(self):
        code, _, _ = run_cli(["-s", example("hdf5/moderate/frames_meta.yaml"),
                              example("hdf5/moderate/frames.h5"), "--index", "99", "-q"])
        assert code == sr.EXIT_DATA


class TestAliasCollisions:
    """The rule that an alias may not shadow a different field."""

    def test_alias_shadowing_a_field_is_rejected(self):
        document = minimal()
        document["aliases"] = {"n": "header.big"}
        document["input"]["header"] = [{"name": "n", "type": "u8"},
                                       {"name": "big", "type": "u8"}]
        with pytest.raises(sr.SchemaError, match="shadows the field"):
            sr.Schema(document, "t.yaml")

    def test_alias_naming_its_own_target_is_allowed(self):
        # bin/complex uses exactly this: an alias that widens where a field can
        # be reached from, without pointing anywhere else.
        document = scoped_schema("$inner")
        document["aliases"] = {"inner": "header.geo.inner"}
        schema = sr.Schema(document, "t.yaml")
        assert schema.index["record.data"].shape[0].refs == ["header.geo.inner"]

    def test_alias_shadowing_a_full_path_is_rejected(self):
        document = minimal("csv")
        document["input"]["fields"] = [{"column": "a"}, {"column": "b"}]
        document["aliases"] = {"a": "b"}
        with pytest.raises(sr.SchemaError, match="shadows the field"):
            sr.Schema(document, "t.yaml")

    def test_alias_shadowing_nrows_is_rejected(self):
        document = minimal("csv")
        document["aliases"] = {"nrows": "a"}
        with pytest.raises(sr.SchemaError, match="shadows the field"):
            sr.Schema(document, "t.yaml")

    def test_empty_alias_table(self):
        document = minimal()
        document["aliases"] = {}
        assert sr.Schema(document, "t.yaml").aliases == {}

    def test_shipped_examples_still_validate(self):
        for path in sorted(glob.glob(os.path.join(EXAMPLES, "**", "*.yaml"), recursive=True)):
            sr.load_schema(path)


class TestOutputConstructs:
    """Output-block features exercised end to end."""

    def test_header_metadata_reaches_the_summary(self, capsys):
        run_cli(["-s", example("bin/complex/scan.yaml"), example("bin/complex/scan.bin"),
                 "-o", os.devnull])
        err = capsys.readouterr().err
        assert "scan v2" in err
        assert "source: example: nested structures" in err
        assert "Multi-plane scan file." in err

    def test_path_declared_in_the_schema(self, tmp_path):
        target = tmp_path / "declared.csv"
        document = load_yaml(example("csv/moderate/sensors_rows.yaml"))
        document["output"]["path"] = str(target)
        path = write_schema(tmp_path, document, "s.yaml")
        code, out, _ = run_cli(["-s", path, example("csv/moderate/sensors.csv"), "-q"])
        assert code == sr.EXIT_OK
        assert out == ""
        assert len(target.read_text(encoding="utf-8").strip().splitlines()) == 5

    def test_alias_expands_inside_an_output_path(self, tmp_path):
        document = load_yaml(example("bin/moderate/capture_frame.yaml"))
        document["aliases"] = {"i": "index"}
        document["output"]["path"] = str(tmp_path / "f_{i}.bin")
        path = write_schema(tmp_path, document, "s.yaml")
        code, _, _ = run_cli(["-s", path, example("bin/moderate/capture.bin"), "-q"])
        assert code == sr.EXIT_OK
        assert sorted(os.listdir(tmp_path)) == ["f_0.bin", "f_1.bin", "f_2.bin", "s.yaml"]

    def test_only_reserved_fields(self, tmp_path):
        document = load_yaml(example("csv/simple/readings.yaml"))
        document["output"] = {"format": "csv", "fields": ["schema", "source", "stem", "index"]}
        path = write_schema(tmp_path, document, "s.yaml")
        code, out, _ = run_cli(["-s", path, example("csv/simple/readings.csv"), "-q"])
        assert code == sr.EXIT_OK
        lines = out.strip().splitlines()
        assert lines[0] == "schema,source,stem,index"
        assert lines[1].split(",") == ["readings", example("csv/simple/readings.csv"),
                                       "readings", "0"]

    def test_format_bin_overriding_a_json_schema(self, tmp_path):
        document = load_yaml(example("csv/moderate/sensors_trace.yaml"))
        document["output"] = {"format": "json", "fields": ["signal"]}
        path = write_schema(tmp_path, document, "s.yaml")
        code, _, payload = run_cli(["-s", path, example("csv/moderate/sensors.csv"),
                                    "--format", "bin", "-q"])
        assert code == sr.EXIT_OK
        assert payload == np.array([0.021, 0.024, 0.019, 0.031], dtype=np.float64).tobytes()


# --------------------------------------------------------------------------
# Every input format combined with every output format in one run
# --------------------------------------------------------------------------

SCALAR_SOURCES = {
    "bin": (example("bin/moderate/capture.bin"), 3),
    "hdf5": (example("hdf5/moderate/frames.h5"), 3),
    "csv": (example("csv/moderate/sensors.csv"), 4),
}

ARRAY_SOURCES = {
    "bin": (example("bin/moderate/capture.bin"), 3, 24),
    "hdf5": (example("hdf5/moderate/frames.h5"), 3, 48),
    "csv": (example("csv/moderate/sensors.csv"), 1, 32),
}


def scalar_schema(tmp_path, fmt: str, out_format: str):
    """A schema over `fmt` emitting the reserved fields plus one scalar named 'value'."""
    if fmt == "bin":
        document = load_yaml(example("bin/moderate/capture_meta.yaml"))
        target = "record.frame_id"
    elif fmt == "hdf5":
        document = load_yaml(example("hdf5/moderate/frames_meta.yaml"))
        target = "stamp"
    else:
        document = load_yaml(example("csv/moderate/sensors_rows.yaml"))
        target = "stamp"
    document["name"] = fmt
    document["output"] = {"format": out_format,
                          "fields": ["schema", "stem", "index",
                                     {"field": target, "as": "value"}]}
    return write_schema(tmp_path, document, f"scalar_{fmt}_{out_format}.yaml")


def array_schema(tmp_path, fmt: str, out_format: str):
    """A schema over `fmt` emitting a single array field named 'payload'."""
    if fmt == "bin":
        document = load_yaml(example("bin/moderate/capture_frame.yaml"))
        target = "record.data"
    elif fmt == "hdf5":
        document = load_yaml(example("hdf5/moderate/frames_image.yaml"))
        target = "frame"
    else:
        document = load_yaml(example("csv/moderate/sensors_trace.yaml"))
        target = "signal"
    document["name"] = fmt
    document["output"] = {"format": out_format,
                          "fields": [{"field": target, "as": "payload"}]}
    return write_schema(tmp_path, document, f"array_{fmt}_{out_format}.yaml")


def groups_for(builder, tmp_path, formats, out_format, sources) -> list[str]:
    """Build the -s SCHEMA INPUT... argument list for the given input formats."""
    argv: list[str] = []
    for fmt in formats:
        argv += ["-s", builder(tmp_path, fmt, out_format), sources[fmt][0]]
    return argv


ALL_FORMATS = ("bin", "hdf5", "csv")


@needs_h5py
class TestFormatMatrix:
    """Several -s groups over unlike input formats, into each output format."""

    def test_three_formats_to_csv(self, tmp_path):
        argv = groups_for(scalar_schema, tmp_path, ALL_FORMATS, "csv", SCALAR_SOURCES)
        code, out, _ = run_cli([*argv, "-q"])
        assert code == sr.EXIT_OK
        rows = list(csvmod.reader(io.StringIO(out)))
        assert rows[0] == ["schema", "stem", "index", "value"]
        assert [row[0] for row in rows[1:]] == ["bin"] * 3 + ["hdf5"] * 3 + ["csv"] * 4
        assert len(rows) == 1 + sum(count for _, count in SCALAR_SOURCES.values())

    def test_three_formats_to_json(self, tmp_path):
        argv = groups_for(scalar_schema, tmp_path, ALL_FORMATS, "json", SCALAR_SOURCES)
        code, out, _ = run_cli([*argv, "-q"])
        assert code == sr.EXIT_OK
        records = json.loads(out)
        assert len(records) == sum(count for _, count in SCALAR_SOURCES.values())
        assert [record["schema"] for record in records] == \
               ["bin"] * 3 + ["hdf5"] * 3 + ["csv"] * 4
        assert all(set(record) == {"schema", "stem", "index", "value"} for record in records)

    def test_three_formats_to_bin_on_stdout(self, tmp_path):
        argv = groups_for(array_schema, tmp_path, ALL_FORMATS, "bin", ARRAY_SOURCES)
        code, text, payload = run_cli([*argv, "-q"])
        assert code == sr.EXIT_OK
        assert text == ""
        assert len(payload) == sum(count * size for _, count, size in ARRAY_SOURCES.values())

    def test_three_formats_to_bin_files(self, tmp_path):
        argv = groups_for(array_schema, tmp_path, ALL_FORMATS, "bin", ARRAY_SOURCES)
        code, _, _ = run_cli([*argv, "-o", str(tmp_path / "out_{stem}_{index}.bin"), "-q"])
        assert code == sr.EXIT_OK
        produced = sorted(name for name in os.listdir(tmp_path) if name.startswith("out_"))
        assert produced == [
            "out_capture_0.bin", "out_capture_1.bin", "out_capture_2.bin",
            "out_frames_1.bin", "out_frames_10.bin", "out_frames_2.bin",
            "out_sensors_0.bin",
        ]
        assert (tmp_path / "out_frames_10.bin").stat().st_size == 48
        assert (tmp_path / "out_sensors_0.bin").stat().st_size == 32

    @pytest.mark.parametrize("out_format", ["json", "csv"])
    def test_several_files_per_group(self, tmp_path, out_format):
        argv: list[str] = []
        for fmt in ALL_FORMATS:
            source = SCALAR_SOURCES[fmt][0]
            argv += ["-s", scalar_schema(tmp_path, fmt, out_format), source, source]
        code, out, _ = run_cli([*argv, "-q"])
        assert code == sr.EXIT_OK
        expected = 2 * sum(count for _, count in SCALAR_SOURCES.values())
        produced = (len(json.loads(out)) if out_format == "json"
                    else len(list(csvmod.reader(io.StringIO(out)))) - 1)
        assert produced == expected

    def test_group_order_is_preserved(self, tmp_path):
        argv = groups_for(scalar_schema, tmp_path, ("csv", "bin", "hdf5"), "csv",
                          SCALAR_SOURCES)
        _, out, _ = run_cli([*argv, "-q"])
        rows = list(csvmod.reader(io.StringIO(out)))
        assert [row[0] for row in rows[1:]] == ["csv"] * 4 + ["bin"] * 3 + ["hdf5"] * 3

    def test_index_applies_across_every_group(self, tmp_path):
        argv = groups_for(scalar_schema, tmp_path, ("bin", "csv"), "json", SCALAR_SOURCES)
        code, out, _ = run_cli([*argv, "--index", "1", "-q"])
        assert code == sr.EXIT_OK
        assert [record["schema"] for record in json.loads(out)] == ["bin", "csv"]

    def test_one_bad_file_in_one_group(self, tmp_path):
        broken = tmp_path / "broken.csv"
        broken.write_text("timestamp,serial_no,ambient\nx,y,z\n", encoding="utf-8")
        argv = groups_for(scalar_schema, tmp_path, ("bin", "csv"), "json", SCALAR_SOURCES)
        argv.append(str(broken))
        code, out, _ = run_cli([*argv, "--skip-errors", "-q"])
        assert code == sr.EXIT_DATA
        assert [record["schema"] for record in json.loads(out)] == ["bin"] * 3 + ["csv"] * 4
        assert run_cli(argv + ["-q"])[0] == sr.EXIT_DATA

    def test_mixed_formats_must_still_agree(self, tmp_path):
        first = scalar_schema(tmp_path, "bin", "csv")
        second = scalar_schema(tmp_path, "csv", "json")
        code, _, _ = run_cli(["-s", first, SCALAR_SOURCES["bin"][0],
                              "-s", second, SCALAR_SOURCES["csv"][0], "-q"])
        assert code == sr.EXIT_SCHEMA

    def test_arrays_from_mixed_formats_refused_by_csv(self, tmp_path):
        argv = groups_for(array_schema, tmp_path, ALL_FORMATS, "bin", ARRAY_SOURCES)
        code, _, _ = run_cli([*argv, "--format", "csv", "-q"])
        assert code == sr.EXIT_USAGE

    def test_shipped_merge_example_covers_three_formats(self):
        code, out, _ = run_cli(["-s", example("merge/legacy.yaml"),
                                example("bin/moderate/capture.bin"),
                                "-s", example("merge/modern.yaml"),
                                example("hdf5/moderate/frames.h5"),
                                "-s", example("merge/tabular.yaml"),
                                example("csv/moderate/sensors.csv"), "-q"])
        assert code == sr.EXIT_OK
        rows = list(csvmod.reader(io.StringIO(out)))
        assert rows[0] == ["schema", "stem", "index", "frame_id", "timestamp"]
        assert [row[0] for row in rows[1:]] == \
               ["legacy"] * 3 + ["modern"] * 3 + ["tabular"] * 4


# --------------------------------------------------------------------------
# 0.4 features
# --------------------------------------------------------------------------

class TestSeekAndRecordSize:
    """Absolute offsets and fixed record slots."""

    @staticmethod
    def catalog(tmp_path, **changes) -> str:
        """The nested catalogue schema, optionally altered."""
        document = load_yaml(example("bin/nested/catalog.yaml"))
        document.update(changes)
        return write_schema(tmp_path, document, "catalog.yaml")

    def test_seek_reaches_the_declared_offset(self):
        _, out, _ = run_cli(["-s", example("bin/nested/catalog.yaml"),
                             example("bin/nested/catalog.bin"), "-q"])
        records = json.loads(out)
        assert [record["record.id"] for record in records] == [1, 2, 3]
        assert records[0]["header.data_offset"] == 48

    def test_seek_past_the_end_is_reported(self, tmp_path):
        path = write_schema(tmp_path, {
            "name": "t",
            "input": {"format": "bin", "header": [
                {"name": "where", "type": "u16"}, {"seek": "$where"},
                {"name": "v", "type": "u8"}]},
        })
        source = pack_bin(tmp_path / "s.bin", struct.pack("<H", 9999) + b"\x01")
        code, _, _ = run_cli(["-s", path, source, "-q"])
        assert code == sr.EXIT_DATA

    def test_seek_backwards(self, tmp_path):
        path = write_schema(tmp_path, {
            "name": "t",
            "input": {"format": "bin", "header": [
                {"name": "a", "type": "u8"}, {"pad": 2}, {"seek": 0},
                {"name": "again", "type": "u8"}]},
        })
        source = pack_bin(tmp_path / "s.bin", bytes([7, 0, 0]))
        code, out, _ = run_cli(["-s", path, source, "-q"])
        assert code == sr.EXIT_OK
        record = json.loads(out)[0]
        assert record["header.a"] == record["header.again"] == 7

    def test_record_size_absorbs_slack(self):
        # Entries hold one to three points but every entry occupies 48 bytes.
        _, out, _ = run_cli(["-s", example("bin/nested/catalog.yaml"),
                             example("bin/nested/catalog.bin"), "-q"])
        assert [record["record.npoints"] for record in json.loads(out)] == [2, 1, 2]

    def test_record_size_enables_direct_access(self, capsys):
        run_cli(["-s", example("bin/nested/catalog.yaml"),
                 example("bin/nested/catalog.bin"), "--index", "2", "-v"])
        assert "seeking directly to record 2" in capsys.readouterr().err

    def test_direct_access_past_the_end(self):
        code, _, _ = run_cli(["-s", example("bin/nested/catalog.yaml"),
                              example("bin/nested/catalog.bin"), "--index", "9", "-q"])
        assert code == sr.EXIT_DATA

    def test_record_larger_than_its_slot(self, tmp_path, capsys):
        path = write_schema(tmp_path, {
            "name": "t",
            "input": {"format": "bin", "record_size": 2, "record_count": 1,
                      "record": [{"name": "v", "type": "u32"}]},
        })
        source = pack_bin(tmp_path / "s.bin", bytes(8))
        code, _, _ = run_cli(["-s", path, source])
        assert code == sr.EXIT_DATA
        assert "more than the declared record_size" in capsys.readouterr().err

    def test_record_size_of_zero(self, tmp_path):
        path = write_schema(tmp_path, {
            "name": "t",
            "input": {"format": "bin", "record_size": 0,
                      "record": [{"name": "v", "type": "u8"}]},
        })
        source = pack_bin(tmp_path / "s.bin", bytes(4))
        assert run_cli(["-s", path, source, "-q"])[0] == sr.EXIT_SCHEMA

    def test_record_size_without_records(self):
        document = minimal()
        document["input"]["record_size"] = 4
        with pytest.raises(sr.SchemaError, match="record_size"):
            sr.Schema(document, "t.yaml")


class TestRepeatingStructures:
    """Arrays of nested structures."""

    def test_whole_structure(self):
        _, out, _ = run_cli(["-s", example("bin/nested/catalog.yaml"),
                             example("bin/nested/catalog.bin"), "--index", "0", "-q"])
        points = json.loads(out)[0]["record.points"]
        assert len(points) == 2
        assert points[0] == {"position": {"x": 0.0, "y": 0.0}, "weight": 2.0}
        assert points[1] == {"position": {"x": 1.0, "y": 2.0}, "weight": 1.0}

    def test_members_become_one_value_per_element(self, tmp_path):
        document = load_yaml(example("bin/nested/catalog.yaml"))
        document["output"] = {"format": "json", "fields": [
            {"field": "record.id", "as": "id"},
            {"field": "record.points.weight", "as": "weights"},
            {"field": "record.points.position.x", "as": "xs"}]}
        path = write_schema(tmp_path, document, "members.yaml")
        code, out, _ = run_cli(["-s", path, example("bin/nested/catalog.bin"), "-q"])
        assert code == sr.EXIT_OK
        records = json.loads(out)
        assert [record["weights"] for record in records] == [[2.0, 1.0], [1.0], [3.0, 2.0]]
        assert records[2]["xs"] == [0.0, 1.0]

    def test_empty_structure_list(self, tmp_path):
        path = write_schema(tmp_path, {
            "name": "t",
            "input": {"format": "bin", "header": [
                {"name": "n", "type": "u8"},
                {"name": "items", "count": "$n", "fields": [{"name": "v", "type": "u8"}]}]},
        })
        source = pack_bin(tmp_path / "s.bin", bytes([0]))
        code, out, _ = run_cli(["-s", path, source, "-q"])
        assert code == sr.EXIT_OK
        assert json.loads(out)[0]["header.items"] == []

    def test_arrays_inside_a_structure(self, tmp_path):
        path = write_schema(tmp_path, {
            "name": "t",
            "input": {"format": "bin", "header": [
                {"name": "n", "type": "u8"},
                {"name": "items", "count": "$n", "fields": [
                    {"name": "buf", "type": "u8", "shape": [2]}]}]},
        })
        source = pack_bin(tmp_path / "s.bin", bytes([2, 1, 2, 3, 4]))
        code, out, _ = run_cli(["-s", path, source, "-q"])
        assert code == sr.EXIT_OK
        items = json.loads(out)[0]["header.items"]
        assert [item["buf"]["data"] for item in items] == [[1, 2], [3, 4]]

    def test_nested_structure_lists(self, tmp_path):
        path = write_schema(tmp_path, {
            "name": "t",
            "input": {"format": "bin", "header": [
                {"name": "outer", "count": 2, "fields": [
                    {"name": "n", "type": "u8"},
                    {"name": "inner", "count": "$n", "fields": [
                        {"name": "v", "type": "u8"}]}]}]},
        })
        source = pack_bin(tmp_path / "s.bin", bytes([2, 10, 11, 1, 12]))
        code, out, _ = run_cli(["-s", path, source, "-q"])
        assert code == sr.EXIT_OK
        outer = json.loads(out)[0]["header.outer"]
        assert [[item["v"] for item in element["inner"]] for element in outer] == \
               [[10, 11], [12]]

    def test_default_output_omits_duplicated_members(self):
        _, out, _ = run_cli(["-s", example("bin/nested/catalog.yaml"),
                             example("bin/nested/catalog.bin"), "--index", "0", "-q"])
        keys = list(json.loads(out)[0])
        assert "record.points" in keys
        assert not any(key.startswith("record.points.") for key in keys)

    @pytest.mark.parametrize("target", ["record.points", "record.points.weight"])
    @pytest.mark.parametrize("out_format", ["csv", "bin", "npy"])
    def test_structures_refused_by_single_valued_outputs(self, tmp_path, target, out_format):
        document = load_yaml(example("bin/nested/catalog.yaml"))
        document["output"] = {"format": out_format, "fields": [target]}
        with pytest.raises(sr.SchemaError):
            sr.Schema(document, "catalog.yaml")

    def test_structure_cannot_be_referenced_by_an_expression(self, tmp_path):
        document = {
            "name": "t",
            "input": {"format": "bin", "header": [
                {"name": "n", "type": "u8"},
                {"name": "items", "count": "$n", "fields": [{"name": "v", "type": "u8"}]},
                {"name": "buf", "type": "u8", "shape": ["$items"]}]},
        }
        with pytest.raises(sr.SchemaError, match="not a single value"):
            sr.Schema(document, "t.yaml")


class TestExpectAndEnum:
    """Magic-number checks and value decoding."""

    def test_magic_matches(self):
        code, _, _ = run_cli(["-s", example("bin/nested/catalog.yaml"),
                              example("bin/nested/catalog.bin"), "-o", os.devnull, "-q"])
        assert code == sr.EXIT_OK

    def test_magic_mismatch_stops_the_read(self, tmp_path, capsys):
        path = write_schema(tmp_path, {
            "name": "t",
            "input": {"format": "bin",
                      "header": [{"name": "magic", "type": "u32", "expect": 0x12345678}]},
        })
        code, _, _ = run_cli(["-s", path, example("bin/nested/catalog.bin")])
        assert code == sr.EXIT_DATA
        assert "expected 305419896" in capsys.readouterr().err

    def test_expect_on_a_string(self, tmp_path):
        path = write_schema(tmp_path, {
            "name": "t",
            "input": {"format": "bin", "header": [
                {"name": "tag", "type": "char", "shape": [4], "expect": "ABCD"}]},
        })
        good = pack_bin(tmp_path / "g.bin", b"ABCD")
        bad = pack_bin(tmp_path / "b.bin", b"WXYZ")
        assert run_cli(["-s", path, good, "-q"])[0] == sr.EXIT_OK
        assert run_cli(["-s", path, bad, "-q"])[0] == sr.EXIT_DATA

    def test_enum_decodes(self):
        _, out, _ = run_cli(["-s", example("bin/nested/catalog.yaml"),
                             example("bin/nested/catalog.bin"), "--index", "0", "-q"])
        assert json.loads(out)[0]["header.kind"] == "image"

    def test_enum_rejects_an_unlisted_value(self, tmp_path, capsys):
        path = write_schema(tmp_path, {
            "name": "t",
            "input": {"format": "bin",
                      "header": [{"name": "k", "type": "u8", "enum": {1: "one"}}]},
        })
        source = pack_bin(tmp_path / "s.bin", bytes([9]))
        code, _, _ = run_cli(["-s", path, source])
        assert code == sr.EXIT_DATA
        assert "not in the enum table" in capsys.readouterr().err

    def test_enum_and_expect_in_csv(self, tmp_path):
        path = write_schema(tmp_path, {
            "name": "t",
            "input": {"format": "csv", "fields": [
                {"column": "state", "type": "u8", "enum": {0: "idle", 1: "busy"}},
                {"column": "tag", "expect": "run"}]},
        })
        data = tmp_path / "d.csv"
        data.write_text("state,tag\n1,run\n0,run\n", encoding="utf-8")
        code, out, _ = run_cli(["-s", path, str(data), "-q"])
        assert code == sr.EXIT_OK
        assert [record["state"] for record in json.loads(out)] == ["busy", "idle"]
        data.write_text("state,tag\n1,other\n", encoding="utf-8")
        assert run_cli(["-s", path, str(data), "-q"])[0] == sr.EXIT_DATA

    @needs_h5py
    def test_expect_and_enum_in_hdf5(self, tmp_path):
        path = write_schema(tmp_path, {
            "name": "t",
            "input": {"format": "hdf5", "fields": [
                {"at": "/@instrument", "expect": "sample-camera"},
                {"at": "/@run_id", "enum": {42: "reference"}}]},
        })
        code, out, _ = run_cli(["-s", path, example("hdf5/simple/instrument.h5"), "-q"])
        assert code == sr.EXIT_OK
        assert json.loads(out)[0]["run_id"] == "reference"

    @pytest.mark.parametrize("entry", [
        {"name": "x", "type": "u8", "shape": [2], "expect": 1},
        {"name": "x", "type": "u8", "shape": [2], "enum": {1: "a"}},
        {"name": "x", "type": "u8", "slice": "[0]"},
        {"name": "x", "type": "u8", "enum": {}},
        {"name": "x", "type": "u8", "enum": {1: ""}},
        {"name": "x", "type": "u8", "expect": [1]},
        {"name": "x", "type": "u8", "shape": [2], "slice": "[q]"},
    ])
    def test_rejections(self, entry):
        document = minimal()
        document["input"]["header"] = [entry]
        with pytest.raises(sr.SchemaError):
            sr.Schema(document, "t.yaml")


class TestSlicing:
    """Reading part of an array."""

    @needs_h5py
    def test_hdf5_slice_is_lazy(self, capsys):
        code, _, payload = run_cli(["-s", example("hdf5/moderate/frames_thumb.yaml"),
                                    example("hdf5/moderate/frames.h5"),
                                    "--index", "1", "--format", "bin", "-q"])
        assert code == sr.EXIT_OK
        assert len(payload) == 2 * 3 * 2      # [::2, ::2] of a 4x6 uint16 frame

    def test_binary_slice(self, tmp_path):
        path = write_schema(tmp_path, {
            "name": "t",
            "input": {"format": "bin", "header": [
                {"name": "buf", "type": "u8", "shape": [6], "slice": "[::2]"}]},
        })
        source = pack_bin(tmp_path / "s.bin", bytes(range(6)))
        _, out, _ = run_cli(["-s", path, source, "-q"])
        assert json.loads(out)[0]["header.buf"]["data"] == [0, 2, 4]

    def test_csv_slice(self, tmp_path):
        path = write_schema(tmp_path, {
            "name": "t",
            "input": {"format": "csv", "fields": [
                {"column": "a", "type": "u8", "shape": ["$nrows"], "slice": "[1:]"}]},
        })
        data = tmp_path / "d.csv"
        data.write_text("a\n1\n2\n3\n", encoding="utf-8")
        _, out, _ = run_cli(["-s", path, str(data), "-q"])
        assert json.loads(out)[0]["a"]["data"] == [2, 3]

    def test_slice_that_cannot_apply(self, tmp_path, capsys):
        path = write_schema(tmp_path, {
            "name": "t",
            "input": {"format": "bin", "header": [
                {"name": "buf", "type": "u8", "shape": [4], "slice": "[0, 0, 0]"}]},
        })
        source = pack_bin(tmp_path / "s.bin", bytes(4))
        code, _, _ = run_cli(["-s", path, source])
        assert code == sr.EXIT_DATA
        assert "slice cannot be applied" in capsys.readouterr().err

    @needs_h5py
    def test_hdf5_slice_that_cannot_apply(self, tmp_path, capsys):
        path = write_schema(tmp_path, {
            "name": "t",
            "input": {"format": "hdf5", "fields": [
                {"at": "/data/frame1/data", "name": "d", "slice": "[0, 0, 0]"}]},
        })
        code, _, _ = run_cli(["-s", path, example("hdf5/moderate/frames.h5")])
        assert code == sr.EXIT_DATA
        assert "slice cannot be applied" in capsys.readouterr().err

    @pytest.mark.parametrize("text", ["[a]", "[1:2:3:4]", "[1:x]", "[1.5]"])
    def test_malformed_slices(self, text):
        with pytest.raises(sr.SchemaError):
            sr.parse_slice(text)

    def test_slice_forms(self):
        assert sr.parse_slice("[:]") == (slice(None),)
        assert sr.parse_slice("[]") == (slice(None),)
        assert sr.parse_slice("[0, ::2]") == (0, slice(None, None, 2))
        assert sr.parse_slice("[..., -1]") == (Ellipsis, -1)


class TestNpyOutput:
    """The self-describing array output format."""

    def test_round_trip_through_numpy(self):
        _, _, payload = run_cli(["-s", example("bin/moderate/capture_frame.yaml"),
                                 example("bin/moderate/capture.bin"),
                                 "--index", "1", "--format", "npy", "-q"])
        array = np.load(io.BytesIO(payload))
        assert array.shape == (3, 4)
        assert array.dtype == np.uint16
        assert array[0].tolist() == [20, 21, 22, 23]

    def test_files_per_record(self, tmp_path):
        code, _, _ = run_cli(["-s", example("bin/moderate/capture_frame.yaml"),
                              example("bin/moderate/capture.bin"), "--format", "npy",
                              "-o", str(tmp_path / "f_{index}.npy"), "-q"])
        assert code == sr.EXIT_OK
        assert sorted(os.listdir(tmp_path)) == ["f_0.npy", "f_1.npy", "f_2.npy"]
        assert np.load(tmp_path / "f_2.npy").shape == (3, 4)

    def test_needs_exactly_one_array(self):
        code, _, _ = run_cli(["-s", example("csv/simple/readings.yaml"),
                              example("csv/simple/readings.csv"), "--format", "npy", "-q"])
        assert code == sr.EXIT_USAGE

    def test_declared_in_the_schema(self, tmp_path):
        document = load_yaml(example("csv/moderate/sensors_trace.yaml"))
        document["output"] = {"format": "npy", "fields": ["signal"]}
        path = write_schema(tmp_path, document, "npy.yaml")
        _, _, payload = run_cli(["-s", path, example("csv/moderate/sensors.csv"), "-q"])
        assert np.load(io.BytesIO(payload)).tolist() == \
               pytest.approx([0.021, 0.024, 0.019, 0.031])


class TestDescribeAndInspect:
    """The two authoring aids."""

    def test_describe_binary(self):
        code, out, _ = run_cli(["-s", example("bin/moderate/capture_meta.yaml"),
                                example("bin/moderate/capture.bin"), "--describe", "-q"])
        assert code == sr.EXIT_OK
        assert "field" in out and "offset" in out
        assert "header.label" in out and "'sample-capture'" in out
        assert "record.data" in out and "<uint16 array [3, 4]>" in out

    def test_describe_uses_the_requested_index(self):
        _, out, _ = run_cli(["-s", example("bin/moderate/capture_meta.yaml"),
                             example("bin/moderate/capture.bin"), "--describe",
                             "--index", "2", "-q"])
        assert "index 2" in out

    def test_describe_csv(self):
        _, out, _ = run_cli(["-s", example("csv/moderate/sensors_rows.yaml"),
                             example("csv/moderate/sensors.csv"), "--describe", "-q"])
        assert "pinned" in out and "serial_no" in out

    @needs_h5py
    def test_describe_hdf5(self):
        _, out, _ = run_cli(["-s", example("hdf5/moderate/frames_meta.yaml"),
                             example("hdf5/moderate/frames.h5"), "--describe",
                             "--index", "2", "-q"])
        assert "/data/frame2@timestamp" in out

    def test_describe_a_missing_file(self):
        code, _, _ = run_cli(["-s", example("csv/simple/readings.yaml"), "/nope.csv",
                              "--describe", "-q"])
        assert code == sr.EXIT_DATA

    def test_inspect_binary(self):
        code, out, _ = run_cli(["--inspect", example("bin/nested/catalog.bin"), "-q"])
        assert code == sr.EXIT_OK
        assert "[bin, 192 byte(s)]" in out
        assert "31 54 41 43" in out          # the CAT1 magic, little-endian on disk
        assert "1TAC" in out

    def test_inspect_csv(self):
        code, out, _ = run_cli(["--inspect", example("csv/simple/readings.csv"), "-q"])
        assert code == sr.EXIT_OK
        assert "delimiter ','" in out
        assert "serial_no" in out

    def test_inspect_csv_guesses_the_delimiter(self):
        code, out, _ = run_cli(["--inspect", example("csv/complex/log.csv"), "-q"])
        assert code == sr.EXIT_OK
        assert "delimiter ';'" in out

    @needs_h5py
    def test_inspect_hdf5(self):
        code, out, _ = run_cli(["--inspect", example("hdf5/moderate/frames.h5"), "-q"])
        assert code == sr.EXIT_OK
        assert "/data/frame10/data" in out
        assert "dataset" in out and "uint16" in out
        assert "/@instrument" in out

    def test_inspect_format_override(self):
        code, out, _ = run_cli(["--inspect", example("csv/simple/readings.csv"),
                                "--inspect-as", "bin", "-q"])
        assert code == sr.EXIT_OK
        assert "[bin," in out

    def test_inspect_several_files(self):
        code, out, _ = run_cli(["--inspect", example("csv/simple/readings.csv"),
                                example("bin/simple/device.bin"), "-q"])
        assert code == sr.EXIT_OK
        assert out.count("byte(s)]") == 2

    def test_inspect_a_missing_file(self):
        code, _, _ = run_cli(["--inspect", "/nope/missing.bin", "-q"])
        assert code == sr.EXIT_DATA

    def test_schema_required_without_inspect(self, capsys):
        code, _, _ = run_cli(["--index", "0"])
        assert code == sr.EXIT_USAGE
        assert "-s SCHEMA" in capsys.readouterr().err


# --------------------------------------------------------------------------
# Generator-independent assertions
#
# Every other binary test compares against values that make_examples.py
# produced. If the generator and the reader shared a misunderstanding about
# byte order or padding, those tests would agree with each other and still be
# wrong. The bytes below are written out by hand.
# --------------------------------------------------------------------------

class TestAgainstLiteralBytes:
    """Values checked against bytes spelled out in the test itself."""

    def test_little_endian_widths(self, tmp_path):
        path = write_schema(tmp_path, {
            "name": "t",
            "input": {"format": "bin", "endian": "little", "header": [
                {"name": "a", "type": "u16"},
                {"name": "b", "type": "i16"},
                {"name": "c", "type": "u32"},
                {"name": "d", "type": "f32"},
            ]},
        })
        source = pack_bin(tmp_path / "s.bin", bytes([
            0x34, 0x12,                    # a = 0x1234
            0xFF, 0xFF,                    # b = -1
            0x78, 0x56, 0x34, 0x12,        # c = 0x12345678
            0x00, 0x00, 0x80, 0x3F,        # d = 1.0
        ]))
        _, out, _ = run_cli(["-s", path, source, "-q"])
        record = json.loads(out)[0]
        assert record["header.a"] == 0x1234
        assert record["header.b"] == -1
        assert record["header.c"] == 0x12345678
        assert record["header.d"] == 1.0

    def test_big_endian_widths(self, tmp_path):
        path = write_schema(tmp_path, {
            "name": "t",
            "input": {"format": "bin", "endian": "big", "header": [
                {"name": "a", "type": "u16"},
                {"name": "b", "type": "i32"},
                {"name": "c", "type": "f64"},
            ]},
        })
        source = pack_bin(tmp_path / "s.bin", bytes([
            0x12, 0x34,                                            # a = 0x1234
            0xFF, 0xFF, 0xFF, 0xFE,                                # b = -2
            0x40, 0x09, 0x21, 0xFB, 0x54, 0x44, 0x2D, 0x18,        # c = pi
        ]))
        _, out, _ = run_cli(["-s", path, source, "-q"])
        record = json.loads(out)[0]
        assert record["header.a"] == 0x1234
        assert record["header.b"] == -2
        assert record["header.c"] == pytest.approx(3.141592653589793)

    def test_padding_is_not_inferred(self, tmp_path):
        # A C compiler would insert a pad byte after the u8; the schema must say
        # so explicitly, and reading without it must produce different values.
        layout = [{"name": "a", "type": "u8"}, {"pad": 1}, {"name": "b", "type": "u16"}]
        padded = write_schema(tmp_path, {
            "name": "t", "input": {"format": "bin", "header": layout}}, "padded.yaml")
        packed = write_schema(tmp_path, {
            "name": "t", "input": {"format": "bin", "header": [layout[0], layout[2]]}},
            "packed.yaml")
        source = pack_bin(tmp_path / "s.bin", bytes([0x01, 0x00, 0x02, 0x00]))
        _, with_pad, _ = run_cli(["-s", padded, source, "-q"])
        _, without, _ = run_cli(["-s", packed, source, "-q"])
        assert json.loads(with_pad)[0]["header.b"] == 2
        # Without the declared pad, b starts one byte early and reads 0x0200.
        assert json.loads(without)[0]["header.b"] == 512

    def test_array_bytes_reach_raw_output_unchanged(self, tmp_path):
        path = write_schema(tmp_path, {
            "name": "t",
            "input": {"format": "bin", "endian": "big",
                      "header": [{"name": "buf", "type": "u16", "shape": [3]}]},
            "output": {"format": "bin", "fields": ["header.buf"]},
        })
        payload = bytes([0x00, 0x01, 0x00, 0x02, 0x00, 0x03])
        source = pack_bin(tmp_path / "s.bin", payload)
        _, _, produced = run_cli(["-s", path, source, "-q"])
        assert produced == payload

    def test_char_stops_at_the_first_nul(self, tmp_path):
        path = write_schema(tmp_path, {
            "name": "t",
            "input": {"format": "bin",
                      "header": [{"name": "tag", "type": "char", "shape": [6]}]},
        })
        source = pack_bin(tmp_path / "s.bin", b"ab\x00cd\x00")
        _, out, _ = run_cli(["-s", path, source, "-q"])
        assert json.loads(out)[0]["header.tag"] == "ab"


# --------------------------------------------------------------------------
# Text encodings and dialects
# --------------------------------------------------------------------------

class TestEncodings:
    """CSV files as other tools actually write them."""

    @staticmethod
    def simple_schema(tmp_path) -> str:
        """A schema reading one numeric column by name."""
        return write_schema(tmp_path, {
            "name": "t",
            "input": {"format": "csv", "fields": [{"column": "a", "type": "f32"}]},
        })

    def test_utf8_bom(self, tmp_path):
        path = self.simple_schema(tmp_path)
        data = tmp_path / "bom.csv"
        data.write_bytes("\ufeffa,b\n1.5,x\n".encode())
        code, out, _ = run_cli(["-s", path, str(data), "-q"])
        assert code == sr.EXIT_OK
        assert json.loads(out)[0]["a"] == 1.5

    def test_crlf_line_endings(self, tmp_path):
        path = self.simple_schema(tmp_path)
        data = tmp_path / "crlf.csv"
        data.write_bytes(b"a,b\r\n1.5,x\r\n2.5,y\r\n")
        code, out, _ = run_cli(["-s", path, str(data), "-q"])
        assert code == sr.EXIT_OK
        assert [record["a"] for record in json.loads(out)] == [1.5, 2.5]

    def test_no_trailing_newline(self, tmp_path):
        path = self.simple_schema(tmp_path)
        data = tmp_path / "bare.csv"
        data.write_bytes(b"a,b\n1.5,x")
        code, out, _ = run_cli(["-s", path, str(data), "-q"])
        assert code == sr.EXIT_OK
        assert len(json.loads(out)) == 1

    def test_latin1_is_reported_not_mangled(self, tmp_path):
        path = self.simple_schema(tmp_path)
        data = tmp_path / "latin.csv"
        data.write_bytes(b"a,b\n1.5,caf\xe9\n")
        code, _, _ = run_cli(["-s", path, str(data), "-q"])
        assert code == sr.EXIT_DATA

    def test_oversized_field_is_reported_cleanly(self, tmp_path, capsys):
        path = self.simple_schema(tmp_path)
        data = tmp_path / "huge.csv"
        data.write_text("a,b\n1.5," + "x" * 200_000 + "\n", encoding="utf-8")
        code, _, _ = run_cli(["-s", path, str(data)])
        err = capsys.readouterr().err
        assert code == sr.EXIT_DATA
        assert "Traceback" not in err
        assert "unexpected" in err


# --------------------------------------------------------------------------
# Fuzzing
#
# The tool takes user-authored schemas and arbitrary files, so the contract
# being tested is that it always exits with one of its documented codes and
# never shows a traceback, whatever it is handed.
# --------------------------------------------------------------------------

VALID_CODES = {sr.EXIT_OK, sr.EXIT_DATA, sr.EXIT_SCHEMA, sr.EXIT_USAGE}


def mutate(document, rng, depth: int = 0):
    """Return a randomly damaged copy of a schema document."""
    junk = [None, 0, -1, 3.5, "", "x", [], {}, [1, 2], {"a": 1}, True, "$", "{index}"]
    if depth > 3:
        return rng.choice(junk)
    if isinstance(document, dict):
        copy = dict(document)
        action = rng.randrange(4)
        if copy and action == 0:
            del copy[rng.choice(list(copy))]
        elif copy and action == 1:
            key = rng.choice(list(copy))
            copy[key] = mutate(copy[key], rng, depth + 1)
        elif action == 2:
            copy[f"bogus{rng.randrange(5)}"] = rng.choice(junk)
        else:
            return rng.choice(junk)
        return copy
    if isinstance(document, list):
        copy = list(document)
        if copy and rng.random() < 0.5:
            position = rng.randrange(len(copy))
            copy[position] = mutate(copy[position], rng, depth + 1)
        else:
            copy.append(rng.choice(junk))
        return copy
    return rng.choice(junk)


class TestFuzzSchemas:
    """Damaged schemas must be reported, never crash."""

    SOURCES: ClassVar[list[tuple[str, str]]] = [
        ("bin/moderate/capture_meta.yaml", "bin/moderate/capture.bin"),
        ("bin/nested/catalog.yaml", "bin/nested/catalog.bin"),
        ("csv/complex/log.yaml", "csv/complex/log.csv"),
    ]

    @pytest.mark.parametrize("seed", range(12))
    def test_mutated_schemas(self, tmp_path, capsys, seed):
        rng = random.Random(seed)
        for schema_path, input_path in self.SOURCES:
            document = mutate(load_yaml(example(schema_path)), rng)
            if not isinstance(document, dict):
                document = {"name": "t", "input": document}
            target = tmp_path / f"fuzz_{seed}.yaml"
            target.write_text(yaml.safe_dump(document, sort_keys=False, default_flow_style=False),
                              encoding="utf-8")
            code, _, _ = run_cli(["-s", str(target), example(input_path), "-q"])
            assert code in VALID_CODES, f"seed {seed} on {schema_path} gave {code}"
            assert "Traceback" not in capsys.readouterr().err

    @pytest.mark.parametrize("text", [
        "", "   ", "[]", "- a\n- b", "name: t", "%YAML 1.2", "\x00\x01",
        "name: t\ninput: {format: bin}", "name: [t]\ninput: null",
        "input:\n  format: bin\n  header: [{name: x, type: u8}]",
    ])
    def test_hand_written_broken_schemas(self, tmp_path, text, capsys):
        target = tmp_path / "broken.yaml"
        target.write_text(text, encoding="utf-8")
        code, _, _ = run_cli(["-s", str(target), example("bin/simple/device.bin"), "-q"])
        assert code in VALID_CODES
        assert "Traceback" not in capsys.readouterr().err


class TestFuzzInput:
    """Damaged input files must be reported, never crash."""

    @pytest.mark.parametrize("seed", range(15))
    def test_mutated_binary(self, tmp_path, capsys, seed):
        rng = random.Random(seed)
        with open(example("bin/nested/catalog.bin"), "rb") as handle:
            payload = bytearray(handle.read())
        for _ in range(rng.randrange(1, 12)):
            payload[rng.randrange(len(payload))] = rng.randrange(256)
        if rng.random() < 0.3:
            del payload[rng.randrange(len(payload)):]
        source = pack_bin(tmp_path / f"fuzz_{seed}.bin", bytes(payload))
        code, _, _ = run_cli(["-s", example("bin/nested/catalog.yaml"), source, "-q"])
        assert code in VALID_CODES, f"seed {seed} gave {code}"
        assert "Traceback" not in capsys.readouterr().err

    @pytest.mark.parametrize("seed", range(10))
    def test_random_bytes_as_input(self, tmp_path, capsys, seed):
        rng = random.Random(seed)
        source = pack_bin(tmp_path / f"noise_{seed}.bin",
                          bytes(rng.randrange(256) for _ in range(rng.randrange(0, 200))))
        for schema_path in ("bin/simple/device.yaml", "bin/complex/scan.yaml",
                            "bin/nested/catalog.yaml"):
            code, _, _ = run_cli(["-s", example(schema_path), source, "-q"])
            assert code in VALID_CODES, f"{schema_path} on seed {seed} gave {code}"
        assert "Traceback" not in capsys.readouterr().err

    @pytest.mark.parametrize("seed", range(10))
    def test_mutated_csv(self, tmp_path, capsys, seed):
        rng = random.Random(seed)
        alphabet = 'ab,;"\n\r\t0123456789.- \\'
        text = "".join(rng.choice(alphabet) for _ in range(rng.randrange(0, 300)))
        source = tmp_path / f"fuzz_{seed}.csv"
        source.write_text(text, encoding="utf-8")
        for schema_path in ("csv/simple/readings.yaml", "csv/complex/log.yaml"):
            code, _, _ = run_cli(["-s", example(schema_path), str(source), "-q"])
            assert code in VALID_CODES, f"{schema_path} on seed {seed} gave {code}"
        assert "Traceback" not in capsys.readouterr().err


# --------------------------------------------------------------------------
# Property-based round-trips
#
# A random schema is generated, a payload is packed to match it with the
# standard library, and the tool must read back exactly what was packed. This
# covers combinations of types, padding, arrays and nesting that no hand-written
# case would enumerate.
# --------------------------------------------------------------------------

ROUNDTRIP_TYPES = {
    "u8": ("B", 0, 255), "i8": ("b", -128, 127),
    "u16": ("H", 0, 65535), "i16": ("h", -32768, 32767),
    "u32": ("I", 0, 2 ** 32 - 1), "i32": ("i", -(2 ** 31), 2 ** 31 - 1),
    "u64": ("Q", 0, 2 ** 63), "i64": ("q", -(2 ** 63), 2 ** 63 - 1),
}


def random_layout(rng, count: int):
    """Build a random flat field list, plus the values it should read back."""
    fields: list[dict] = []
    expected: dict[str, Any] = {}
    codes: list[str] = []
    values: list[Any] = []
    for position in range(count):
        roll = rng.random()
        if roll < 0.15:
            width = rng.randrange(1, 4)
            fields.append({"pad": width})
            codes.append(f"{width}x")
        elif roll < 0.35:
            length = rng.randrange(1, 6)
            name = f"f{position}"
            fields.append({"name": name, "type": "u8", "shape": [length]})
            data = [rng.randrange(256) for _ in range(length)]
            codes.append(f"{length}B")
            values.extend(data)
            expected[name] = data
        elif roll < 0.45:
            length = rng.randrange(2, 8)
            name = f"f{position}"
            text = "".join(rng.choice("abcdefgh") for _ in range(rng.randrange(1, length)))
            fields.append({"name": name, "type": "char", "shape": [length]})
            codes.append(f"{length}s")
            values.append(text.encode())
            expected[name] = text
        else:
            name = f"f{position}"
            type_name = rng.choice(list(ROUNDTRIP_TYPES))
            code, low, high = ROUNDTRIP_TYPES[type_name]
            value = rng.randrange(low, high)
            fields.append({"name": name, "type": type_name})
            codes.append(code)
            values.append(value)
            expected[name] = value
    return fields, codes, values, expected


class TestRoundTrip:
    """Random schemas packed with struct and read back by the tool."""

    @pytest.mark.parametrize("seed", range(25))
    def test_flat_layout(self, tmp_path, seed):
        rng = random.Random(seed)
        endian = rng.choice(["little", "big"])
        prefix = "<" if endian == "little" else ">"
        fields, codes, values, expected = random_layout(rng, rng.randrange(1, 9))
        path = write_schema(tmp_path, {
            "name": "t", "input": {"format": "bin", "endian": endian, "header": fields},
        }, f"rt_{seed}.yaml")
        source = pack_bin(tmp_path / f"rt_{seed}.bin",
                          struct.pack(prefix + "".join(codes), *values))
        code, out, _ = run_cli(["-s", path, source, "-q"])
        assert code == sr.EXIT_OK, f"seed {seed}"
        record = json.loads(out)[0]
        for name, value in expected.items():
            produced = record[f"header.{name}"]
            if isinstance(produced, dict):
                produced = produced["data"]
            assert produced == value, f"seed {seed}, field {name}"

    @pytest.mark.parametrize("seed", range(15))
    def test_records_round_trip(self, tmp_path, seed):
        rng = random.Random(seed)
        fields, codes, _, _ = random_layout(rng, rng.randrange(1, 5))
        count = rng.randrange(1, 5)
        packer = struct.Struct("<" + "".join(codes))
        payload = b""
        expected_records = []
        for _ in range(count):
            values: list[Any] = []
            expected: dict[str, Any] = {}
            for entry in fields:
                if "pad" in entry:
                    continue
                name = entry["name"]
                if entry["type"] == "char":
                    width = entry["shape"][0]
                    text = "".join(rng.choice("abcd") for _ in range(rng.randrange(1, width)))
                    values.append(text.encode())
                    expected[name] = text
                elif "shape" in entry:
                    data = [rng.randrange(256) for _ in range(entry["shape"][0])]
                    values.extend(data)
                    expected[name] = data
                else:
                    _, low, high = ROUNDTRIP_TYPES[entry["type"]]
                    value = rng.randrange(low, high)
                    values.append(value)
                    expected[name] = value
            payload += packer.pack(*values)
            expected_records.append(expected)
        path = write_schema(tmp_path, {
            "name": "t",
            "input": {"format": "bin", "endian": "little", "record_count": count,
                      "record": fields},
        }, f"rec_{seed}.yaml")
        source = pack_bin(tmp_path / f"rec_{seed}.bin", payload)
        code, out, _ = run_cli(["-s", path, source, "-q"])
        assert code == sr.EXIT_OK, f"seed {seed}"
        produced = json.loads(out)
        assert len(produced) == count
        for record, expected in zip(produced, expected_records, strict=True):
            for name, value in expected.items():
                got = record[f"record.{name}"]
                if isinstance(got, dict):
                    got = got["data"]
                assert got == value, f"seed {seed}, field {name}"

    @pytest.mark.parametrize("seed", range(10))
    def test_csv_round_trip(self, tmp_path, seed):
        rng = random.Random(seed)
        rows = [[rng.randrange(-1000, 1000) for _ in range(3)]
                for _ in range(rng.randrange(1, 6))]
        data = tmp_path / f"rt_{seed}.csv"
        data.write_text("a,b,c\n" + "\n".join(",".join(str(cell) for cell in row)
                                              for row in rows) + "\n", encoding="utf-8")
        path = write_schema(tmp_path, {
            "name": "t",
            "input": {"format": "csv", "fields": [
                {"column": "a", "type": "i32"}, {"column": "b", "type": "i32"},
                {"column": "c", "type": "i32"}]},
        }, f"rt_{seed}.yaml")
        code, out, _ = run_cli(["-s", path, str(data), "-q"])
        assert code == sr.EXIT_OK
        produced = [[record["a"], record["b"], record["c"]] for record in json.loads(out)]
        assert produced == rows

    @pytest.mark.parametrize("seed", range(8))
    def test_raw_output_matches_the_source_bytes(self, tmp_path, seed):
        rng = random.Random(seed)
        length = rng.randrange(1, 40)
        payload = bytes(rng.randrange(256) for _ in range(length * 2))
        path = write_schema(tmp_path, {
            "name": "t",
            "input": {"format": "bin", "endian": "big",
                      "header": [{"name": "buf", "type": "u16", "shape": [length]}]},
            "output": {"format": "bin", "fields": ["header.buf"]},
        }, f"raw_{seed}.yaml")
        source = pack_bin(tmp_path / f"raw_{seed}.bin", payload)
        code, _, produced = run_cli(["-s", path, source, "-q"])
        assert code == sr.EXIT_OK
        assert produced == payload


# --------------------------------------------------------------------------
# Memory
# --------------------------------------------------------------------------

class TestMemory:
    """The claim that reads stream rather than buffer, made testable."""

    def test_skipped_arrays_are_never_materialised(self, tmp_path):
        source = tmp_path / "large.bin"
        TestScale.build(source, 64, 512)          # 32 MB of pixel data
        path = write_schema(tmp_path, load_yaml(example("bin/moderate/capture_meta.yaml")),
                            "meta.yaml")
        tracemalloc.start()
        try:
            code, _, _ = run_cli(["-s", path, str(source), "-o", str(tmp_path / "o.csv"), "-q"])
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert code == sr.EXIT_OK
        # Reading 64 frames of scalars must not approach the 32 MB of pixels.
        assert peak < 4_000_000, f"peak allocation was {peak} bytes"

    def test_one_frame_costs_one_frame(self, tmp_path):
        source = tmp_path / "large.bin"
        TestScale.build(source, 64, 512)
        path = write_schema(tmp_path, load_yaml(example("bin/moderate/capture_frame.yaml")),
                            "frame.yaml")
        tracemalloc.start()
        try:
            code, _, _ = run_cli(["-s", path, str(source), "--index", "63",
                                  "-o", str(tmp_path / "one.bin"), "-q"])
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert code == sr.EXIT_OK
        # One 512x512 uint16 frame is 512 kB; a few copies of it is the ceiling.
        assert peak < 3_000_000, f"peak allocation was {peak} bytes"


class TestRemainingBranches:
    """Rarely-taken paths in the 0.4 features."""

    def test_enum_key_must_be_a_number_or_text(self):
        document = minimal()
        document["input"]["header"] = [{"name": "k", "type": "u8", "enum": {1.5: "a"}}]
        with pytest.raises(sr.SchemaError, match="usable raw value"):
            sr.Schema(document, "t.yaml")

    def test_record_size_running_past_the_end(self, tmp_path):
        # No record_count, so records are read to end of file; the last slot is
        # short and is dropped rather than half-read.
        path = write_schema(tmp_path, {
            "name": "t",
            "input": {"format": "bin", "record_size": 4,
                      "record": [{"name": "v", "type": "u8"}]},
        })
        source = pack_bin(tmp_path / "s.bin", bytes([1, 0, 0, 0, 2, 0]))
        code, out, _ = run_cli(["-s", path, source, "-q"])
        assert code == sr.EXIT_OK
        assert [record["record.v"] for record in json.loads(out)] == [1, 2]

    def test_padding_inside_a_repeating_structure(self, tmp_path):
        path = write_schema(tmp_path, {
            "name": "t",
            "input": {"format": "bin", "header": [
                {"name": "n", "type": "u8"},
                {"name": "items", "count": "$n", "fields": [
                    {"name": "v", "type": "u8"}, {"pad": 1}]}]},
            "output": {"format": "json", "fields": ["header.items.v"]},
        })
        source = pack_bin(tmp_path / "s.bin", bytes([2, 7, 0, 8, 0]))
        code, out, _ = run_cli(["-s", path, source, "-q"])
        assert code == sr.EXIT_OK
        assert json.loads(out)[0]["header.items.v"] == [7, 8]

    def test_describe_shows_a_structure_summary(self):
        _, out, _ = run_cli(["-s", example("bin/nested/catalog.yaml"),
                             example("bin/nested/catalog.bin"), "--describe", "-q"])
        assert "record.points" in out
        assert "item(s)>" in out

    def test_inspect_stops_after_a_few_rows(self, tmp_path):
        data = tmp_path / "long.csv"
        data.write_text("a,b\n" + "".join(f"{n},{n}\n" for n in range(50)), encoding="utf-8")
        code, out, _ = run_cli(["--inspect", str(data), "-q"])
        assert code == sr.EXIT_OK
        assert "column" in out

    def test_inspect_an_unreadable_csv(self, tmp_path):
        code, _, _ = run_cli(["--inspect", str(tmp_path), "--inspect-as", "csv", "-q"])
        assert code == sr.EXIT_DATA

    def test_inspect_an_empty_csv(self, tmp_path):
        data = tmp_path / "empty.csv"
        data.write_text("", encoding="utf-8")
        code, out, _ = run_cli(["--inspect", str(data), "-q"])
        assert code == sr.EXIT_OK
        assert "(no rows)" in out

    def test_csv_enum_rejects_an_unlisted_value(self, tmp_path):
        path = write_schema(tmp_path, {
            "name": "t",
            "input": {"format": "csv",
                      "fields": [{"column": "s", "type": "u8", "enum": {0: "idle"}}]},
        })
        data = tmp_path / "d.csv"
        data.write_text("s\n7\n", encoding="utf-8")
        assert run_cli(["-s", path, str(data), "-q"])[0] == sr.EXIT_DATA


@needs_h5py
class TestHdf5RemainingBranches:
    """HDF5 paths reached only by unusual files."""

    @pytest.fixture()
    def vectors(self, tmp_path) -> str:
        """A file whose root attribute holds a vector."""
        path = tmp_path / "vec.h5"
        with h5py.File(path, "w") as handle:
            handle.attrs["corners"] = np.arange(6, dtype=np.uint16)
            handle.attrs["state"] = np.uint8(2)
            handle.attrs["label"] = "alpha"
        return str(path)

    def test_slice_on_an_attribute_is_rejected(self, tmp_path, vectors):
        # An attribute is declared as a single value, so there is nothing to
        # slice; slicing applies to datasets and to declared array fields.
        path = write_schema(tmp_path, {
            "name": "t",
            "input": {"format": "hdf5", "fields": [{"at": "/@corners", "slice": "[::2]"}]},
        })
        assert run_cli(["-s", path, vectors, "-q"])[0] == sr.EXIT_SCHEMA

    def test_array_attribute_reads_as_an_array(self, tmp_path, vectors):
        path = write_schema(tmp_path, {
            "name": "t", "input": {"format": "hdf5", "fields": [{"at": "/@corners"}]},
        })
        code, out, _ = run_cli(["-s", path, vectors, "-q"])
        assert code == sr.EXIT_OK
        assert json.loads(out)[0]["corners"]["data"] == [0, 1, 2, 3, 4, 5]

    def test_expect_mismatch(self, tmp_path, vectors):
        path = write_schema(tmp_path, {
            "name": "t",
            "input": {"format": "hdf5", "fields": [{"at": "/@label", "expect": "beta"}]},
        })
        assert run_cli(["-s", path, vectors, "-q"])[0] == sr.EXIT_DATA

    def test_enum_rejects_an_unlisted_value(self, tmp_path, vectors):
        path = write_schema(tmp_path, {
            "name": "t",
            "input": {"format": "hdf5", "fields": [{"at": "/@state", "enum": {1: "one"}}]},
        })
        assert run_cli(["-s", path, vectors, "-q"])[0] == sr.EXIT_DATA

    def test_slice_without_brackets(self, tmp_path):
        assert sr.parse_slice("::2") == (slice(None, None, 2),)
        path = write_schema(tmp_path, {
            "name": "t",
            "input": {"format": "bin", "header": [
                {"name": "buf", "type": "u8", "shape": [4], "slice": "1:3"}]},
        })
        source = pack_bin(tmp_path / "s.bin", bytes([9, 8, 7, 6]))
        _, out, _ = run_cli(["-s", path, source, "-q"])
        assert json.loads(out)[0]["header.buf"]["data"] == [8, 7]

    def test_record_size_with_a_count_and_an_index(self, tmp_path):
        path = write_schema(tmp_path, {
            "name": "t",
            "input": {"format": "bin", "record_size": 4, "record_count": 2,
                      "record": [{"name": "v", "type": "u8"}]},
        })
        source = pack_bin(tmp_path / "s.bin", bytes([1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0]))
        _, out, _ = run_cli(["-s", path, source, "--index", "1", "-q"])
        assert json.loads(out)[0]["record.v"] == 2
        assert run_cli(["-s", path, source, "--index", "2", "-q"])[0] == sr.EXIT_DATA
