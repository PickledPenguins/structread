# structread

**Version 0.5**

Extract values and arrays from binary, HDF5 and CSV files, using YAML schemas
that describe both the input structure and the output structure.

A schema says what an input file looks like and where the interesting parts are;
it also says what should come out and in what form. Several schemas can be
combined in one run, so files with different internal structures can be merged
into a single table or document as long as they agree on what the output looks
like.

---

## Contents

**[Part I — Using structread](#part-i--using-structread)**

- [What it does](#what-it-does)
- [Quick start](#quick-start)
- [Installation](#installation)
- [The command line](#the-command-line)
- [Writing a schema](#writing-a-schema)
- [Types](#types)
- [Choosing the output](#choosing-the-output)
- [Reserved fields](#reserved-fields)
- [Worked examples](#worked-examples)
- [Example gallery](#example-gallery)
- [Exit status](#exit-status)

**[Part II — Advanced usage](#part-ii--advanced-usage)**

- [Expressions](#expressions)
- [Names, scope and references](#names-scope-and-references)
- [Aliases](#aliases)
- [Repeating structures](#repeating-structures)
- [Absolute offsets and fixed record slots](#absolute-offsets-and-fixed-record-slots)
- [Validating a file as it is read](#validating-a-file-as-it-is-read)
- [Slicing](#slicing)
- [Several schemas in one run](#several-schemas-in-one-run)
- [Several input files](#several-input-files)
- [Missing and unrepresentable values](#missing-and-unrepresentable-values)
- [Diagnostics and logging](#diagnostics-and-logging)
- [Describing and inspecting](#describing-and-inspecting)
- [Rules enforced at schema load](#rules-enforced-at-schema-load)
- [Assumptions and limitations](#assumptions-and-limitations)

**[Part III — Developer reference](#part-iii--developer-reference)**

- [Architecture](#architecture)
- [Module layout](#module-layout)
- [The schema model](#the-schema-model)
- [The expression engine](#the-expression-engine)
- [Name resolution](#name-resolution)
- [The reader interface](#the-reader-interface)
- [The writer interface](#the-writer-interface)
- [Run orchestration](#run-orchestration)
- [Error taxonomy](#error-taxonomy)
- [Debugging a run](#debugging-a-run)
- [Performance notes](#performance-notes)
- [Adding a format](#adding-a-format)
- [Project structure](#project-structure)
- [Development and testing](#development-and-testing)
- [Future work](#future-work)
- [Changelog](#changelog)

---

# Part I — Using structread

## What it does

```
   schema A ─┬─▶ files a1 a2 a3 ─┐
             │                   │
   schema B ─┴─▶ files b1 b2 ────┤   identical output structures
                                 │
                                 ▼
                  json  ·  csv  ·  bin  ·  npy
```

Three input formats: `bin`, `hdf5`, `csv`.
Four output formats: `json`, `csv`, `bin` (raw bytes) and `npy` (numpy's own
self-describing array file).

Two rules keep the combinations sane:

- **Multi-valued fields never go to CSV.** A CSV cell holds one value, so arrays
  and repeating structures are rejected there.
- **`bin` and `npy` output carry exactly one array field.** An array file has no
  room for labels, so scalars have no meaning in it.

Everything else is permitted: scalars to CSV or JSON, arrays to JSON, raw bytes
or npy, from any input format.

With no `output` block and no command-line override, every field defined in
`input` is written as JSON to standard output, keyed by full field path.

## Quick start

Read every frame's metadata from a directory of captures into one spreadsheet:

```
structread -s capture.yaml runs/*.h5 -o summary.csv
```

Pull one frame out and pipe the raw bytes into another program:

```
structread -s capture.yaml run01.h5 --index 7 --format bin | display-tool
```

Merge two differently-structured file sets into one table:

```
structread -s old_format.yaml legacy/*.bin -s new_format.yaml current/*.h5 -o all.csv
```

See everything a schema extracts, as JSON on the terminal:

```
structread -s capture.yaml run01.h5
```

Look at a file before writing a schema for it:

```
structread --inspect run01.h5
```

Check a schema without touching any input:

```
structread -s capture.yaml --validate
```

## Installation

### Requirements

| Component | Version | Needed for |
|---|---|---|
| Python | 3.10 or later | everything |
| numpy | 1.20 or later | everything |
| PyYAML | 5.1 or later | everything |
| h5py | 3.0 or later | `format: hdf5` only |

`h5py` is imported only when an HDF5 schema is used. Without it, HDF5 schemas
fail with exit status 4 and a message naming the package; binary and CSV schemas
are unaffected. CSV is handled by the standard library.

### Installing

Copy the project directory anywhere and run the launcher, which resolves its own
location and works from any working directory:

```
./structread --version
ln -s /opt/structread/structread ~/bin/structread
```

### Without network access

On a connected host of the same platform and Python version:

```
pip download numpy pyyaml h5py -d structread-wheels
```

On the target host:

```
pip install --no-index --find-links structread-wheels numpy pyyaml h5py
```

## The command line

```
usage: structread [-h] -s SCHEMA INPUT... [-s SCHEMA INPUT...] [options]
       structread --inspect FILE... [--inspect-as FORMAT]

Extract values and arrays from binary, HDF5 and CSV files using YAML schemas.

options:
  -h, --help            show this help message and exit
  -s SCHEMA INPUT [SCHEMA INPUT ...], --schema SCHEMA INPUT [SCHEMA INPUT ...]
                        a schema followed by the input files it reads; repeatable
  --inspect FILE [FILE ...]
                        describe what is in these files without a schema, then exit
  --inspect-as FORMAT   treat --inspect files as this format instead of guessing
  -o PATH, --output PATH
                        destination; '-' means standard output. May contain {stem} and
                        {index} for bin output
  --format {json,csv,bin,npy}
                        override the output format of every schema
  --index SPEC          record, frame or row to read: an integer or 'all' (default: all)
  --indent N            JSON indentation, 0 for compact (default: 2)
  --skip-errors         continue past files that cannot be read
  --validate            validate the schemas and exit without reading input
  --describe            show how each schema resolves against its input - field locations,
                        sizes and values - then exit
  -q, --quiet           log warnings and errors only
  -v, --verbose         log at debug level; repeat for per-field tracing
  --log-file PATH       also write a debug-level log to this file
  --version             show program's version number and exit

Exit status: 0 success, 1 data error, 2 schema error, 3 usage error, 4 missing dependency.
```

| Option | Argument | Default | Effect |
|---|---|---|---|
| `-s`, `--schema` | schema then inputs | — | Starts a group: the first path is a YAML schema, every path after it is an input file read with that schema. Repeatable. |
| `--inspect` | paths | — | Show what is inside these files without a schema, then exit. Used on its own, without `-s`. |
| `--inspect-as` | `bin`\|`hdf5`\|`csv` | guessed from the filename | Force the format used by `--inspect`. |
| `-o`, `--output` | path | `output.path`, else standard output | Destination. `-` forces standard output even when the schema names a file. May contain `{stem}` and `{index}` for `bin` and `npy` output. |
| `--format` | `json`\|`csv`\|`bin`\|`npy` | `output.format`, else `json` | Overrides the output format of every schema. The output rules are still enforced. |
| `--index` | `all` or an integer | `all` | Which record, frame or row to read. |
| `--indent` | integer ≥ 0 | `2` | JSON indentation; `0` produces a single compact line. |
| `--skip-errors` | — | off | Continue past a file that cannot be read, reporting it as a warning. |
| `--validate` | — | off | Validate every schema and exit without reading input. |
| `--describe` | — | off | Show how each schema resolves against its input — every field's location, size and value — then exit. |
| `-q`, `--quiet` | — | off | Log warnings and errors only. |
| `-v`, `--verbose` | — | off | Log at debug level. Repeat (`-vv`) for per-field tracing. |
| `--log-file` | path | — | Also write the log to this file, always at trace level regardless of console verbosity. |
| `--version` | — | — | Print the version and exit. |
| `-h`, `--help` | — | — | Print usage and exit. |

Log output always goes to standard error, never to standard output, so raw
binary output can be piped safely. A consumer that stops reading — `head`, for
instance — closes the pipe, which is treated as a normal end rather than a
failure.

## Writing a schema

Every schema has the same outer shape:

```yaml
name: capture                    # required
version: 3                       # optional
source: "ICD-4471 rev C, §5.2"   # optional; where the layout is documented
description: >                   # optional
  Frame capture files written by the acquisition system.
aliases: {n: index}              # optional; see Part II
input: {...}                     # required
output: {...}                    # optional
```

| Key | Required | Meaning |
|---|---|---|
| `name` | yes | Identifies the schema. Appears in log output and in error messages. |
| `version` | no | Version of the *file format* this schema describes. Informational; logged at startup. |
| `source` | no | Where the layout came from — a document, ticket or specification reference. Informational. |
| `description` | no | Free text. The first line is logged at startup. |
| `aliases` | no | Short names for use in paths and expressions. |
| `input` | yes | What the file looks like. Format-specific; see below. |
| `output` | no | What should come out. Defaults to JSON on standard output. |

The top-level `source` key describes the schema's provenance. It is unrelated to
the reserved output field also called `source`, which holds an input filename;
one is a schema key, the other a field name, and they never appear in the same
position.

Validation is strict: an unrecognised key is an error rather than being ignored,
so a misspelling is reported instead of silently doing nothing.

### Binary input

```yaml
input:
  format: bin
  endian: little           # little | big | native   (default: little)

  header:                  # optional; read once, at offset 0
    - {name: magic,   type: u32}
    - {name: version, type: u16}
    - {name: nframes, type: u16}
    - {name: label,   type: char, shape: [16]}

  record_count: $header.nframes   # optional
  record_size: 48                 # optional
  record:                         # optional; repeats after the header
    - {name: frame_id,  type: u64}
    - {name: width,     type: u16}
    - {name: height,    type: u16}
    - {pad: 2}
    - {name: timestamp, type: f64}
    - {name: data,      type: u16, shape: [$height, $width]}
```

| Key | Required | Meaning |
|---|---|---|
| `format` | yes | `bin`. |
| `endian` | no | Byte order for every multi-byte field: `little`, `big` or `native`. |
| `header` | no | Fields read once from offset 0. In scope for every record and every expression. |
| `record` | no | Fields read repeatedly, immediately after the header. |
| `record_count` | no | Number of records, as an integer or an expression over header fields. Omitted: records are read until the file is exhausted. Requires `record`. |
| `record_size` | no | Byte size of each record slot. See [Absolute offsets and fixed record slots](#absolute-offsets-and-fixed-record-slots). Requires `record`. |

At least one of `header` and `record` must be present.

A field entry accepts:

| Key | Meaning |
|---|---|
| `name` | Field name. Required. |
| `type` | A type name from the table below. Required. |
| `shape` | Present for arrays and strings, absent for scalars. **Required for arrays and for `char`.** |
| `expect` | A constant the field must hold; a mismatch stops the read. Single values only. |
| `enum` | A mapping of raw value to name; the name replaces the number in the output. Single values only. |
| `slice` | A numpy-style index applied to an array after it is read, e.g. `"[::2, ::2]"`. Arrays only. |

A structure entry accepts:

| Key | Meaning |
|---|---|
| `name` | Structure name. Required. |
| `fields` | The fields it contains. Required. Opens a new scope. |
| `count` | Makes the structure repeat — see [Repeating structures](#repeating-structures). |

Two directives may appear anywhere a field may, each used on its own:

| Directive | Meaning |
|---|---|
| `pad` | Skip this many bytes forward. |
| `seek` | Move to an absolute offset, given as an integer or an expression. |

Padding is never inferred. Compilers insert alignment padding that varies by
target and by pragma, so it is declared explicitly and a schema states exactly
what is on disk.

### HDF5 input

`fields` is a list; each entry names one location.

```yaml
aliases: {n: index}

input:
  format: hdf5
  fields:
    - {at: "/@run_id",                    type: u32}
    - {at: "/@instrument",                type: str}
    - {at: "/data/frame{n}@exposure_us",  type: u32}
    - {at: "/data/frame{n}@timestamp",    name: stamp, type: f64}
    - {at: "/data/frame{n}/data@rows",    name: rows,  type: u16}
    - {at: "/data/frame{n}/data@cols",    name: cols,  type: u16}
    - {at: "/data/frame{n}/data",         name: frame, type: u16, shape: [$rows, $cols]}
```

| Key | Required | Meaning |
|---|---|---|
| `format` | yes | `hdf5`. |
| `fields` | yes | The locations to read, described below. |

A field entry accepts:

| Key | Required | Meaning |
|---|---|---|
| `at` | yes | Location. `@name` selects an attribute of the object to its left; without `@`, the location is a dataset. `{index}` — or an alias for it — is replaced by the current index. |
| `name` | no | Field name. **Defaults to the attribute name, or to the dataset's last path segment.** Required when that default would not be a legal identifier. |
| `type` | no | Expected type. A mismatch is an error rather than a silent conversion. Omitted: the file's own type is used, and the choice is logged. |
| `shape` | no | Expected shape, checked against the dataset. `shape: []` declares a 0-dimensional dataset, which is then treated as a single value. |
| `slice` | no | A numpy-style index applied to the dataset as it is read, not to the result. Datasets only. |
| `expect` | no | A constant the value must hold. Single values only. |
| `enum` | no | A mapping of raw value to name. Single values only. |

`{index}` may appear anywhere in a path, including inside a name segment
(`/scan/img_{index}_raw`). With `--index all`, matching paths are discovered by
substituting a number pattern and are visited in ascending numeric order, so
`frame2` precedes `frame10`.

Fields not named in `output.fields`, and not needed by an expression, are not
read at all.

### CSV input

`fields` is a list of columns.

```yaml
input:
  format: csv
  header: true             # first line holds column names (default: true)
  delimiter: ","           # default: ","
  skip_lines: 0            # lines discarded before the header (default: 0)

  fields:
    - {column: timestamp}                              # name "timestamp", type str
    - {column: serial_no, name: serial}                # renamed
    - {column: ambient,   type: f32}                   # typed
    - {column: signal,    type: f32, shape: [$nrows]}  # whole column as an array
    - {columns: [x, y, z], name: xyz, type: f32}       # several columns, 2-D array
```

| Key | Required | Meaning |
|---|---|---|
| `format` | yes | `csv`. |
| `header` | no | Whether the first line holds column names. Default `true`. |
| `delimiter` | no | The field separator, a single character. Default `,`. |
| `skip_lines` | no | Lines discarded before the header, for files that start with comments. Default `0`. |
| `fields` | yes | The columns to read, described below. |

A field entry accepts:

| Key | Required | Meaning |
|---|---|---|
| `column` | one of `column`/`columns` | Column by header name, or by zero-based position when an integer is given. |
| `columns` | one of `column`/`columns` | Several columns, producing a 2-D array with one row per data row. |
| `name` | no | Field name. **Defaults to the value of `column:` when that is a name.** Required for a positional column and for `columns`. |
| `type` | no | Cell type. **Defaults to `str`**, which is stated per field in the startup summary so the assumption is visible. |
| `row` | no | Pin this field to one data row. Without it, a scalar field reads the row selected by the index. Cannot be combined with `shape` or `columns`. |
| `shape` | no | For a whole-column array; validated against the number of data rows. |
| `slice` | no | A numpy-style index applied to a whole-column array after it is read. |
| `expect` | no | A constant every cell must hold. |
| `enum` | no | A mapping of raw value to name. |

A field with `shape` or `columns` is an array and reads the whole column. A field
with neither is a scalar and reads a single cell.

`$nrows` holds the number of data rows and is available to CSV expressions.

Blank lines are ignored. A row shorter than the columns a schema asks for
produces a warning and reads the missing cells as empty. A UTF-8 byte-order mark
is stripped. Input must be UTF-8; anything else is reported rather than mangled.

A column name that is not a legal identifier — one containing spaces, dots or
punctuation — must be given a `name:`, and the error says so.

## Types

| Name | Width | Notes |
|---|---|---|
| `i8` `i16` `i32` `i64` | 1, 2, 4, 8 | Signed integers. |
| `u8` `u16` `u32` `u64` | 1, 2, 4, 8 | Unsigned integers. |
| `f32` `f64` | 4, 8 | IEEE 754 floating point. |
| `bool` | 1 | Binary: any non-zero byte is true. CSV: `1/true/yes/y/t` and `0/false/no/n/f`, case-insensitive. |
| `char` | 1 per element | Binary only. With `shape: [N]`, N bytes decoded as UTF-8 and cut at the first NUL. Produces a string, not an array. |
| `str` | — | HDF5 and CSV only. Text as stored. The CSV default. |

`shape` is a list of dimensions, each an integer or an expression:

```yaml
shape: [1024, 1024]
shape: [$height, $width]
shape: [$nrows]
shape: [$header.geometry.height, $header.geometry.width, 3]
```

| Format | Role of `shape` |
|---|---|
| `bin` | **Required for arrays.** Nothing else says where the array ends. |
| `hdf5` | **Optional.** The dataset carries its own shape; a declared shape is checked and a mismatch is an error. `[]` declares a scalar dataset. |
| `csv` | **Optional.** Checked against the number of data rows. |

Declaring a shape already known is a cheap way to make "wrong file" fail loudly
instead of producing plausible nonsense.

## Choosing the output

```yaml
output:
  format: csv              # json | csv | bin | npy   (default: json)
  path: summary.csv        # default: standard output
  fields:
    - stem
    - index
    - record.frame_id
    - {field: record.timestamp, as: t}    # renamed in the output
```

| Key | Required | Meaning |
|---|---|---|
| `format` | no | Output format. Overridden by `--format`. |
| `path` | no | Destination. Overridden by `-o`. May contain `{stem}` and `{index}` for `bin` and `npy` output. |
| `fields` | no | Which fields to emit, in this order. Defaults to every field in `input`, in declaration order, keyed by full path. |

Each entry in `fields` is either a full field path, or a mapping:

| Key | Required | Meaning |
|---|---|---|
| `field` | yes | The full path of the field to emit. |
| `as` | no | The name to use in the JSON key or CSV column header. Defaults to the path. |

Renaming is what lets two schemas with different internal layouts agree on one
output structure. Duplicate output names are rejected.

Omitting the whole block is the same as writing `output: {format: json}`.

None of the shipped example schemas declare a `path:`, so running them writes to
standard output and leaves no files behind.

### JSON

Scalars appear directly. Arrays appear as an object with the values flattened in
row-major order:

```json
[
  {
    "stem": "run01",
    "index": 1,
    "run_id": 42,
    "frame": {
      "dtype": "uint16",
      "shape": [3, 4],
      "data": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    }
  }
]
```

```python
frame = np.array(record["frame"]["data"]).reshape(record["frame"]["shape"])
```

A flat list plus `shape` is used rather than nested lists because it is far
smaller for real image data and needs no recursive walk to read back. Text
arrays report `"dtype": "str"`.

### CSV

```
stem,index,run_id,timestamp
run01,1,42,1001.0
run01,2,42,1002.0
run02,1,43,2001.0
```

Column order follows `output.fields` exactly; the header uses each field's `as:`
name where given, otherwise its full path. Values containing the delimiter, a
quote or a newline are quoted per RFC 4180.

A schema sending a multi-valued field to CSV is rejected at load time. An HDF5
attribute that turns out to hold a vector is caught when it is read, since
nothing in the file declares that in advance.

### Raw bin

The array's bytes exactly as stored: no header, no separator, no padding,
row-major, in the array's own byte order and element width. Nothing else reaches
standard output, so the stream pipes directly.

`output.fields` must name exactly one field and that field must be an array.
Element type and shape are the caller's responsibility; they are visible at any
time by running the same schema with `--format json` or `--describe`.

### npy

The same single array, written in numpy's own format. The file carries dtype,
shape and byte order in its header, so it reloads without the schema:

```
$ structread -s examples/hdf5/moderate/frames_thumb.yaml examples/hdf5/moderate/frames.h5 --index 1 -o thumb.npy -q
$ python3 -c "import numpy; print(numpy.load('thumb.npy').shape)"
(2, 3)
```

Use `bin` when another program expects a headerless buffer, and `npy` when the
data has to survive on its own.

## Reserved fields

Always available in `output.fields` and in output paths, alongside the fields
from `input`:

| Name | Value |
|---|---|
| `source` | The input path exactly as given on the command line |
| `stem` | The filename with directory and extension removed |
| `index` | The record, frame or row index this output row came from |
| `schema` | The `name:` of the schema that read this file — useful when merging groups |

They appear only when listed. A schema defining an input field with one of these
names is rejected.

## Worked examples

Every command below runs against the shipped gallery from the project root.

### 1. See everything a schema extracts

```
$ structread -s examples/bin/moderate/capture_meta.yaml examples/bin/moderate/capture.bin --format json --index 0 -q
[
  {
    "stem": "capture",
    "index": 0,
    "label": "sample-capture",
    "frame_id": 1,
    "width": 4,
    "height": 3,
    "timestamp": 1001.0
  }
]
```

### 2. Metadata to a spreadsheet

```
$ structread -s examples/bin/moderate/capture_meta.yaml examples/bin/moderate/capture.bin -q
stem,index,label,frame_id,width,height,timestamp
capture,0,sample-capture,1,4,3,1001.0
capture,1,sample-capture,2,4,3,1002.0
capture,2,sample-capture,3,4,3,1003.0
```

The default index of `all` produced one row per record. `record.data` is not in
the field list, so no pixel data was read.

### 3. One frame as raw bytes

```
$ structread -s examples/bin/moderate/capture_frame.yaml examples/bin/moderate/capture.bin --index 1 -q | od -A d -t u2
0000000    20    21    22    23    24    25    26    27
0000016    28    29    30    31
0000024
```

### 4. Every frame to its own file

```
$ structread -s examples/bin/moderate/capture_frame.yaml examples/bin/moderate/capture.bin -o "frame_{index}.bin" -q
$ ls
frame_0.bin  frame_1.bin  frame_2.bin
```

Without a placeholder, the second frame would overwrite the first, so the run is
refused:

```
ERROR   usage error: more than one array would be written to all.bin; add
        {index} or {stem} to the output path, or select one record with --index N
```

### 5. HDF5 attributes across a set of files

```
$ structread -s examples/hdf5/moderate/frames_meta.yaml examples/hdf5/moderate/frames.h5 -q
stem,index,run_id,instrument,stamp,exposure
frames,1,42,sample-camera,1001.0,250
frames,2,42,sample-camera,1002.0,500
frames,10,42,sample-camera,1010.0,2500
```

The file holds `frame1`, `frame2` and `frame10`; `index` carries the real frame
number, and the ordering is numeric rather than lexical.

### 6. Merging three file formats into one table

```
$ structread -s examples/merge/legacy.yaml examples/bin/moderate/capture.bin -s examples/merge/modern.yaml examples/hdf5/moderate/frames.h5 -s examples/merge/tabular.yaml examples/csv/moderate/sensors.csv -q
schema,stem,index,frame_id,timestamp
legacy,capture,0,1,1001.0
legacy,capture,1,2,1002.0
legacy,capture,2,3,1003.0
modern,frames,1,250,1001.0
modern,frames,2,500,1002.0
modern,frames,10,2500,1010.0
tabular,sensors,0,0.021,1001.0
tabular,sensors,1,0.024,1002.0
tabular,sensors,2,0.019,1003.0
tabular,sensors,3,0.031,1004.0
```

All three schemas emit the same five names from three different file formats;
where the values come from is each schema's own business.

### 7. CSV rows, with a pinned field and an empty cell

```
$ structread -s examples/csv/moderate/sensors_rows.yaml examples/csv/moderate/sensors.csv -q
stem,index,serial,stamp,ambient
sensors,0,SN-4471,1001.0,21.5
sensors,1,SN-4471,1002.0,21.6
sensors,2,SN-4471,1003.0,21.4
sensors,3,SN-4471,1004.0,
```

### 8. Whole CSV columns as arrays

```
$ structread -s examples/csv/moderate/sensors_trace.yaml examples/csv/moderate/sensors.csv -q --indent 0
[{"serial": "SN-4471", "signal": {"dtype": "float64", "shape": [4], "data": [0.021, 0.024, 0.019, 0.031]}, "xyz": {"dtype": "float32", "shape": [4, 3], "data": [1.0, 2.0, 3.0, 1.5, 2.5, 3.5, 2.0, 3.0, 4.0, 2.5, 3.5, 4.5]}}]
```

One result, not four: no field varies with the index, because `serial` is pinned
to a row and the other two read whole columns.

### 9. A record holding an array of structures

```
$ structread -s examples/bin/nested/catalog.yaml examples/bin/nested/catalog.bin --index 1 -q --indent 0
[{"header.magic": 1128354865, "header.version": 2, "header.kind": "image", "header.entry_count": 3, "header.data_offset": 48, "header.description": "sample catalogue", "record.id": 2, "record.label": "entry-2", "record.npoints": 1, "record.points": [{"position": {"x": 0.0, "y": 0.0}, "weight": 1.0}]}]
```

### 10. Continuing past a bad file

```
$ structread -s meta.yaml runs/*.h5 --skip-errors
WARNING runs/run07.h5: attribute 'run_id' not found on /; file skipped
INFO    wrote 11 record(s) to standard output
WARNING 1 file(s) skipped
$ echo $?
1
```

Every readable file still contributes. Without `--skip-errors` the run stops at
the first failure and writes nothing.

## Example gallery

`examples/` holds ten input files with matching schemas, three or four per
format, at increasing complexity, plus a merge set. Together they exercise every
schema feature. All are small, and the test suite reads each one and converts it
to every output format it legally supports, so the gallery cannot drift out of
date. Each directory has its own README explaining what it shows.

| Example | Demonstrates |
|---|---|
| `bin/simple` | Header only, fixed scalars, `char` string, `pad`, `bool`, default JSON output. |
| `bin/moderate` | Counted records, a shape sized from record fields, output renaming, metadata to CSV and pixels to raw bin from one file. |
| `bin/nested` | A repeating structure holding a structure, `seek:` to an offset the header declares, `record_size:` fixed slots, `expect:` as a magic check, `enum:` decoding. |
| `bin/complex` | Nesting three levels deep, an alias across scopes, big-endian, every signed width, a 3-D array, an arithmetic shape, a shape mixing a literal with an expression, a zero-length dimension, variable-length records read to end of file. |
| `hdf5/simple` | Root attributes, names defaulted from attribute names, declared types as validation. |
| `hdf5/moderate` | `{index}` paths, the `n: index` alias, numeric ordering across `frame2`/`frame10`, declared type and shape checked against a dataset, lazy slicing, `npy` output. |
| `hdf5/complex` | `{index}` inside a name segment, shape resolved from sibling attributes, a scalar dataset via `shape: []`, metadata from a parallel group. |
| `csv/simple` | Names defaulted from columns, the `str` type default and how it is reported. |
| `csv/moderate` | Renamed columns, declared types, a row-pinned scalar, a single-column array, a multi-column 2-D array, an empty cell. |
| `csv/complex` | No header row, a semicolon delimiter, `skip_lines`, positional columns, quoted fields with delimiters, newlines and escaped quotes, empty cells. |
| `merge` | Three schemas over all three input formats producing one table. |

## Exit status

| Code | Meaning |
|---|---|
| 0 | Everything requested was read and written. |
| 1 | A data error: a file could not be read, a value was missing, a declared type or shape did not match. Also returned when `--skip-errors` tolerated at least one failure. |
| 2 | A schema error: malformed YAML, an unknown key, a broken reference, or a rule from Part II. |
| 3 | A usage error: arguments that cannot be acted on. |
| 4 | A library needed for an input format is not installed. |

Messages name the file, the field and the location:

```
ERROR   capture.bin: record 2: field record.data needs 16 byte(s) at offset 156
        but only 8 remain
ERROR   run01.h5: attribute 'exposure_us' not found on /data/frame7
ERROR   run09.h5: /data/frame2/data has shape [8, 6], schema declares [4, 6]
ERROR   sensors.csv: row 4, field ambient: cannot read 'n/a' as f32
        (could not convert string to float: 'n/a')
ERROR   other.bin: field header.magic at offset 0: expected 1128354865,
        found 1398035012
ERROR   capture.bin: record 2 consumed 52 byte(s), more than the declared
        record_size of 48
ERROR   schema error: schema mismatch: 'legacy' outputs csv, 'modern' outputs
        json; all schemas in one run must agree, or use --format
```

---

# Part II — Advanced usage

## Expressions

Sizes, counts and offsets accept small arithmetic expressions:

```
$header.nframes
$width * $height
($rows - 1) * $stride
0x40
```

- Operators `+`, `-`, `*`, `/`, parentheses and unary minus. Decimal or
  hexadecimal literals.
- `$name` follows the scope rules below and may be a short name or a full path.
- Only fields already read may be referenced; a reference to a later field, or
  to the field being defined, is rejected at schema load.
- Only single values may be referenced. Arrays and repeating structures cannot
  be used in arithmetic.
- Division yields an integer when exact, otherwise a fraction; a fractional or
  negative result where a whole number is required is an error.
- Expressions are parsed by a dedicated tokeniser and recursive-descent parser.
  Nothing in a schema reaches `eval()`, so a schema file cannot execute code.

Expressions may appear in `shape`, `count`, `record_count`, `record_size` and
`seek`.

## Names, scope and references

Fields live in a tree. A binary schema has `header` and `record` at the top;
either may contain nested structures, each of which opens a new scope.

```yaml
input:
  format: bin
  header:
    - {name: magic,   type: u32}
    - {name: nframes, type: u16}
    - name: geometry              # a nested structure: opens scope header.geometry
      fields:
        - {name: width,  type: u16}
        - {name: height, type: u16}
  record:
    - {name: frame_id, type: u64}
    - {name: data, type: u16, shape: [$header.geometry.height, $header.geometry.width]}
```

The full name of a field is its path from the root: `header.nframes`,
`header.geometry.width`, `record.data`.

Candidates for a reference are gathered in this order:

1. The current scope.
2. Each enclosing scope in turn, working outward.
3. The full path from the root.

If exactly one candidate exists, it is used. If more than one exists the
reference is **ambiguous and the schema is rejected**, with every candidate
named, rather than one being chosen silently:

```
ERROR schema error: capture.yaml: schema.input.header[3].shape: $outer is
      ambiguous - it could mean header.geo.outer or header.outer; write the
      full path
```

So from inside `header.geometry`, `$width` finds `header.geometry.width`, and
`$nframes` finds `header.nframes` because `header` encloses it. From inside
`record`, neither is visible by short name — `record` does not enclose `header` —
so the full path is required.

A full path always works, in every scope. Writing `$header.geometry.width` from
inside `header.geometry` is legal and is the clearer choice in a large schema.

Names appear in three places: in expressions as `$name`; in `output.fields`
always as the full path, so the field list is unambiguous however the input tree
is arranged; and in log output, always as the full path. For HDF5 and CSV, whose
fields are flat, the full path is just the field name.

## Aliases

`aliases` maps a short name to a full one. Aliases apply to `$references` and to
`{}` placeholders in HDF5 paths and output paths.

```yaml
aliases:
  n: index
  planes: header.geometry.planes

input:
  format: hdf5
  fields:
    - {at: "/data/frame{n}/data", type: u16}
```

An alias may match a whole reference (`$planes`) or its leading segment
(`$geo.inner` with `geo: header.geometry`).

Rules, all checked at schema load: alias names must be legal identifiers,
aliases may not chain to other aliases, an alias may not shadow a reserved field
name, and every alias must ultimately resolve to a real field or to `index`.

An alias may not shadow a **field** either. Aliases are expanded before
references are resolved, so an alias sharing a name with a field would win
silently — the opposite of the ambiguity rule that governs every other name:

```yaml
aliases: {n: header.big}     # rejected: 'n' is also a field name
header:
  - {name: n,   type: u8}
  - {name: big, type: u8}
```

An alias that names the very field it points at is allowed, since it only widens
where that field can be reached from. `examples/bin/complex` uses exactly that:
`planes: header.geometry.planes` makes `$planes` usable from inside `record`,
where the short name would otherwise not resolve.

## Repeating structures

A structure with a `count:` repeats, producing an array of nested structures.
The count is an expression, so it may come from a field read moments earlier:

```yaml
  record:
    - {name: npoints, type: u16}
    - name: points
      count: $npoints
      fields:
        - name: position          # a structure inside each element
          fields:
            - {name: x, type: f32}
            - {name: y, type: f32}
        - {name: weight, type: f32}
```

The structure is emitted whole:

```json
"record.points": [
  {"position": {"x": 0.0, "y": 0.0}, "weight": 2.0},
  {"position": {"x": 1.0, "y": 2.0}, "weight": 1.0}
]
```

Its members are addressable too, each yielding one value per element, which is
how one column of the list is pulled out on its own:

```yaml
output:
  fields:
    - {field: record.points.weight, as: weights}
```

```json
{"weights": [2.0, 1.0]}
```

Rules:

- Structures may nest and repeat to any depth, including a repeating structure
  inside a repeating structure.
- A count of zero yields an empty list; it is not an error.
- Expressions may not reference a repeating structure or its members, since
  those are not single values.
- Anything from a repeating structure is multi-valued, so it goes to JSON only —
  never to CSV, `bin` or `npy`.
- With no `output.fields`, the structure is emitted whole and its members are
  left out, since they would only repeat what the structure already carries.

## Absolute offsets and fixed record slots

`pad:` moves forward by a fixed number of bytes; `seek:` moves to an absolute
offset, which is how a header that declares where its data begins is followed:

```yaml
  header:
    - {name: data_offset, type: u32}
    - {name: description, type: char, shape: [24]}
    - {seek: $data_offset}     # jump to where the header says the data starts
```

A `seek:` may appear anywhere a field may, and may move backwards as well as
forwards. Seeking past the end of the file is an error.

`record_size:` declares that every record occupies a fixed slot:

```yaml
  record_count: $header.entry_count
  record_size: 48
```

Two things follow. Slack after the declared fields is skipped, so a record that
holds one item and a record that holds three both advance by 48 bytes. And
record N is reached directly rather than by walking the records before it, which
turns `--index 99999` from a linear walk into a seek.

A record whose fields consume more than the declared slot is an error. A final
slot that is short is still emitted if all of its declared fields were present,
with a warning that the slot itself was incomplete.

## Validating a file as it is read

Four mechanisms turn "this is the wrong file" into an immediate error rather
than plausible-looking nonsense.

**`expect:`** — a constant the value must hold. This is the cheapest sanity
check available and is worth putting on any magic number:

```yaml
    - {name: magic, type: u32, expect: 0x43415431}
```

```
ERROR other.bin: field header.magic at offset 0: expected 1128354865,
      found 1398035012
```

`expect` works on single values in all three input formats, and on `char`
strings, where it compares text.

**`enum:`** — a mapping of raw value to name. The name replaces the number in
the output, and a value the table does not list is an error rather than a
silent passthrough:

```yaml
    - {name: kind, type: u8, enum: {1: image, 2: spectrum}}
```

**Declared `type`** — for HDF5 and CSV, where the file carries its own types,
declaring one turns a mismatch into an error.

**Declared `shape`** — likewise for HDF5 datasets and CSV column arrays. For
binary arrays the shape is instruction rather than validation, since nothing
else says where the array ends.

## Slicing

`slice:` takes a numpy-style index string and reads only part of an array:

```yaml
    - {at: "/data/frame{n}/data", name: thumb, type: u16, slice: "[::2, ::2]"}
```

Accepted forms are integers, `start:stop:step` with any part omitted, `...`, and
any combination of those separated by commas. The brackets are optional. The
string is parsed by hand, so nothing in a schema reaches `eval()`.

For HDF5 the slice is applied to the dataset rather than to the result, so a
large frame is never read in full just to be decimated. For binary and CSV the
array is read and then sliced, since the bytes have to be traversed anyway.

A slice that cannot apply to the array's shape is an error.

## Several schemas in one run

`-s` starts a group. The first path after it is the schema; every following path
is an input file read with that schema, until the next `-s` or the end of the
command line.

```
structread -s y1.yaml f1 f2 f3 -s y2.yaml f4 f5 -o combined.csv
```

`f1 f2 f3` are read with `y1.yaml`; `f4 f5` with `y2.yaml`. Results are appended
in command-line order into one output.

**All schemas in a run must agree on the output.** Specifically:

| Element | Requirement |
|---|---|
| `output.format` | Identical across every schema, unless `--format` overrides them all. |
| `output.fields` | The same output names in the same order. |
| `output.path` | Either identical, declared in only one schema, or supplied with `-o`. Conflicting paths are an error. |

The field *names* must match; the fields they are wired to need not. This is the
point of the feature: a legacy binary layout and a modern HDF5 layout can both
produce a `timestamp` column from completely different places. See
`examples/merge/` for a working trio.

A group with a schema but no input files is an error, since it would silently do
nothing. `--validate` is the exception, since it reads no input at all.

## Several input files

Any number of input paths may follow each `-s`. Results combine according to the
output format:

| Output format | Combination |
|---|---|
| `json` | One JSON array of objects, one per (file × index). A single file at a single index still yields an array of one. |
| `csv` | One row per (file × index), all in one file, header written once. |
| `bin`, `npy` | Streams concatenated in argument order. To standard output that is one stream; with `-o`, a second array bound for a path already written is refused rather than overwriting it, so the path needs `{stem}` or `{index}` unless the run produces exactly one array. |

Which file a row came from is recorded only if `source`, `stem` or `schema`
appears in `output.fields`. Nothing is added automatically.

`--index` means one idea in three spellings, and **defaults to `all`**:

| Input format | `--index N` selects | `--index all` (default) covers |
|---|---|---|
| `bin` | Record N after the header | `record_count` records, or every record until end of file |
| `hdf5` | The path with `{index}` replaced by N | Every path matching `{index}`, in ascending numeric order |
| `csv` | Data row N | Every data row |

Fields that do not depend on the index — a binary header field, an HDF5 root
attribute, a CSV field pinned with `row:` — repeat unchanged across every index.
A schema with no index-dependent fields yields exactly one result per file, and
`--index all` and `--index 0` behave identically.

## Missing and unrepresentable values

| Situation | Result |
|---|---|
| A field not present in the values read | `null` in JSON, an empty cell in CSV |
| An empty CSV cell in a text field | `""` — text has no missing value |
| An empty CSV cell in a numeric scalar | `null` in JSON, an empty cell in CSV |
| An empty CSV cell in an `f32`/`f64` array | `null` in JSON — the array holds a NaN |
| An empty CSV cell in an integer array | An error, since an integer array cannot represent it. The message suggests `f32` or `f64` |
| A non-finite float from any source | `null` in JSON, an empty cell in CSV |
| A CSV row shorter than the schema expects | A warning, then the missing cells read as empty |

## Diagnostics and logging

Logging uses the standard `logging` module, writes to standard error, and never
touches standard output.

| Level | Shown by default | Contents |
|---|---|---|
| ERROR | yes | Failures that stop the run. |
| WARNING | yes | Tolerated problems: a file skipped under `--skip-errors`, trailing bytes, a short record slot, a short CSV row. |
| INFO | yes | The startup summary, one line per file read, and the final count. |
| DEBUG (`-v`) | no | Per-record progress and sizes, discovered indices, direct seeks, schema loading. |
| TRACE (`-vv`) | no | Per-field reads with byte offsets and values, seeks, and every field skipped because the output does not need it. |

`-q` silences INFO and below. `--log-file PATH` records everything at trace level
regardless of console verbosity, so a failing run can be re-examined without
being repeated.

### The startup summary

Printed before any file is opened, so every assumption and default is visible up
front:

```
INFO    structread 0.5
INFO    schema    capture_meta v1 (examples/bin/moderate/capture_meta.yaml)
INFO              Per-frame metadata from a capture file, as a table.
INFO    input     bin, endian=little
INFO              header: 4 field(s)
INFO              record: 6 field(s)
INFO              record_count: $header.nframes
INFO    files     examples/bin/moderate/capture.bin
INFO    index     all [default]
INFO    output    csv -> captures.csv
INFO              fields: stem, index, label, frame_id, width, height, timestamp
INFO    assumed   [capture_meta] not in output, will not be read: header.magic,
                  header.version, record.data
```

The `assumed` block names every default that was applied rather than declared.
For CSV it is usually the largest section, because untyped columns default to
`str`:

```
INFO    assumed   [readings] csv header=true [default]
INFO    assumed   [readings] csv delimiter="," [default]
INFO    assumed   [readings] csv skip_lines=0 [default]
INFO    assumed   [readings] field "timestamp": name from column header
INFO    assumed   [readings] field "ambient": type str [default] - declare
                  type: f32 or similar for numeric output
INFO    assumed   [readings] no output block: json to standard output [default]
INFO    assumed   [readings] no output.fields: every field emitted, keyed by full path
```

### Per-file logging

```
INFO    examples/bin/moderate/capture.bin: 162 bytes, header read (24 bytes consumed)
INFO    examples/bin/moderate/capture.bin: record_count = 3
DEBUG   examples/bin/moderate/capture.bin: record 0 at offset 24, size 46
TRACE   examples/bin/moderate/capture.bin: record.width = 4 at offset 32
TRACE   examples/bin/moderate/capture.bin: record.data skipped (24 bytes, not in output)
INFO    wrote 3 record(s) to captures.csv
```

## Describing and inspecting

Two modes exist to help write a schema rather than run one.

### `--describe`: how a schema resolves against a file

```
$ structread -s examples/bin/moderate/capture_meta.yaml examples/bin/moderate/capture.bin --describe -q
examples/bin/moderate/capture.bin  [schema capture_meta, index 0]
field             type  offset  bytes  value
----------------  ----  ------  -----  ---------------------
header.magic      u32   0       4      1398035012
header.version    u16   4       2      1
header.nframes    u16   6       2      3
header.label      char  8       16     'sample-capture'
record.frame_id   u64   24      8      1
record.width      u16   32      2      4
record.height     u16   34      2      3
record.timestamp  f64   38      8      1001.0
record.data       u16   46      24     <uint16 array [3, 4]>
```

Every field is shown, including ones the output does not use, with the byte
offset it was actually read from. `--index N` describes that record instead.
For HDF5 the table shows each field's resolved location; for CSV it shows the
column each field reads and whether it is pinned to a row.

### `--inspect`: what is in a file, with no schema at all

```
$ structread --inspect examples/hdf5/moderate/frames.h5 -q
examples/hdf5/moderate/frames.h5  [hdf5, 13096 byte(s)]
location                  kind       type    shape or value
------------------------  ---------  ------  ---------------
/@instrument              attribute  str     'sample-camera'
/@run_id                  attribute  int     42
/data                     group
/data/frame1              group
/data/frame1@exposure_us  attribute  int     250
/data/frame1/data         dataset    uint16  [4, 6]
```

For CSV it reports the delimiter it guessed, the columns and the first rows; for
anything else it prints the opening bytes as hex and text, which is usually
enough to find a magic number:

```
$ structread --inspect examples/bin/nested/catalog.bin -q
examples/bin/nested/catalog.bin  [bin, 192 byte(s)]
00000000  31 54 41 43 02 00 01 00 03 00 00 00 30 00 00 00  1TAC........0...
```

The format is guessed from the filename — `.h5`, `.hdf5` and `.he5` are HDF5;
`.csv`, `.tsv` and `.txt` are CSV; anything else is binary — and can be forced
with `--inspect-as`.

## Rules enforced at schema load

Checked by `--validate`, and at the start of every run before any input is
opened:

- `output.format: csv` with a multi-valued field in `output.fields` → error.
- `output.format: bin` or `npy` with anything other than exactly one array field
  → error.
- A field in `output.fields` that is neither defined in `input` nor reserved →
  error, listing the available fields.
- A nested structure that is not a value named in `output.fields` → error.
- Duplicate output names → error.
- An `input` field named `source`, `stem`, `index` or `schema` → error.
- Two fields resolving to the same path → error.
- A field name that is not a legal identifier, including one defaulted from a
  column or attribute → error, quoting the offending name.
- A binary array field, or a `char` field, without `shape` → error.
- A reference with more than one candidate under the scope rules → error,
  listing every candidate.
- A reference to an unknown field, a later field, or a field that is not a
  single value → error.
- An alias that shadows a reserved name or a field name, chains to another
  alias, or does not resolve → error. An alias naming the field it points at is
  allowed.
- `record_count` or `record_size` without a `record` block → error.
- A binary schema with neither `header` nor `record` → error.
- A CSV field combining `row:` with `shape:` or `columns:` → error, since `row`
  selects one cell and the others read every row.
- `expect:` or `enum:` on anything but a single value, or `slice:` on anything
  but an array → error.
- An enum table that is empty, or whose keys are not numbers or text, or whose
  names are empty → error.
- A field from a repeating structure sent to CSV, `bin` or `npy` → error.
- A malformed slice string → error.
- Across a multi-schema run: differing `output.format`, differing output field
  names, or conflicting `output.path` → error, showing both schemas' values.
- A `-s` group with no input files → error.
- An unknown type name, endianness, format, or any unrecognised key → error.

## Assumptions and limitations

- One run produces one output. Extracting metadata to CSV and frames to raw bin
  from the same file means two invocations.
- Every schema in a run must agree on the output structure; the inputs may differ
  freely.
- Binary record lengths are fixed or derivable from fields already read. Records
  delimited by a terminator sentinel are not supported.
- Binary input must be a seekable regular file; standard input cannot be a
  binary source.
- Reaching binary record N means walking the records before it, unless the schema
  declares `record_size`, in which case record N is reached directly. Array data
  in skipped records is not read either way.
- Expressions may reference only fields already read, and only single values.
- Alignment padding is never inferred and must be declared with `pad:`.
- Bit fields are not addressable; the smallest addressable unit is one byte.
- HDF5 compound and variable-length dataset types are not converted.
- An HDF5 attribute holding a vector is treated as an array, which a CSV output
  reports as an error when it is read rather than at schema load, since nothing
  declares it in advance.
- CSV arrays are whole columns; arbitrary rectangular cell ranges are not
  supported.
- A CSV file is read fully into memory; binary and HDF5 reads are streamed.
- Raw bin output carries no type or shape information; that lives in the schema
  and can be seen with `--describe`. Use `npy` when the data must be
  self-describing.
- Anything from a repeating structure is multi-valued and can only go to JSON.
- `--inspect` guesses a file's format from its name unless told otherwise.
- Output files are replaced, not appended to.

---

# Part III — Developer reference

## Architecture

One pass, four stages:

```
  YAML ──▶ Schema ──▶ Reader ──▶ Writer ──▶ output
           (validate  (per input  (per output
            once)      format)     format)
```

`Schema` is built and fully validated before any input file is opened, so a
schema error costs nothing and a run that starts will not stop for a schema
reason. A `Reader` is chosen by `input.format` and yields `(index, values)`
pairs, where `values` maps full field paths to Python scalars, strings, numpy
arrays or lists. A `Writer` is chosen by `output.format` and consumes one record
at a time. `execute()` drives the loop across every `-s` group.

Three design rules run through the module:

- **Nothing from a schema is executed.** Expressions and slice strings are
  parsed by hand rather than passed to `eval()`, so a schema file is inert data.
- **Unknown keys are errors.** Every mapping is checked against an explicit key
  list, so a misspelled option is reported instead of ignored.
- **Ambiguity is an error.** Where a name could resolve two ways, the schema is
  rejected rather than one interpretation being chosen silently.

## Module layout

Six modules, each a layer of the pipeline. The dependency graph is acyclic and
one-directional: every arrow points down this table, so a module can only use
what is above it.

| Module | Lines | Responsibility | Imports from |
|---|---|---|---|
| `core.py` | ~220 | Errors and exit codes, logging and the `TRACE` level, the type table, value conversion, and the constant tables naming what the tool accepts. | — |
| `expressions.py` | ~270 | The expression tokeniser and parser, `Expression`, and the numpy-style slice parser. | `core` |
| `schema.py` | ~680 | `Field`, `OutputSpec`, `Schema` with one builder per input format, and `load_schema`. All validation lives here. | `core`, `expressions` |
| `readers.py` | ~580 | `BinReader`, `Hdf5Reader`, `CsvReader` and the `READERS` table. | `core`, `expressions`, `schema` |
| `writers.py` | ~190 | `JsonWriter`, `CsvWriter`, `BinWriter` and `make_writer`. | `core`, `schema` |
| `structread.py` | ~600 | The front end: run orchestration, `--describe` and `--inspect`, argument parsing, and the exception-to-exit-code mapping. Re-exports the public names, so `import structread` reaches the whole tool. | all of the above |

The split follows the pipeline rather than file size: a change to how a format
is read touches `readers.py` alone, a change to what a schema may say touches
`schema.py` alone, and a change to how the command behaves touches
`structread.py` alone.

The key sets the validator accepts are constants in `core.py` —
`SCHEMA_KEYS_REQUIRED`, `SCHEMA_KEYS_OPTIONAL`, `BIN_INPUT_KEYS`,
`CSV_INPUT_KEYS`, `HDF5_INPUT_KEYS`, `BIN_FIELD_KEYS`, `BIN_STRUCT_KEYS`,
`BIN_DIRECTIVES`, `HDF5_FIELD_KEYS`, `CSV_FIELD_KEYS`, `OUTPUT_KEYS`,
`OUTPUT_FIELD_KEYS` — so the vocabulary the tool accepts can be read in one
place instead of being reconstructed from a dozen call sites.

## The schema model

`Field` is the single node type. Its `kind` decides everything downstream:

| Kind | Meaning | Emitted as |
|---|---|---|
| `scalar` | One number, boolean or string | the value |
| `string` | A `char` array decoded to text | the text |
| `array` | A numpy array | an array, or raw bytes |
| `list` | A repeating structure | a list of items |
| `struct` | A non-repeating structure | nothing itself; its members are separate fields |
| `pad` | Skip forward | nothing |
| `seek` | Move to an offset | nothing |
| `pseudo` | A value the reader supplies, currently only CSV's `nrows` | usable in expressions |

`Field.is_value` is true for the first four, and is what `leaf_paths()` and the
output validator filter on. `Schema.index` maps every full path to its `Field`;
`Schema.roots` holds the top-level nodes in declaration order; `Field.order` is a
monotonic counter used to reject forward references.

`OutputSpec` holds the resolved output format, path and `(field path, output
name)` pairs. `OutputSpec.signature()` is what `check_agreement` compares across
a multi-schema run.

## The expression engine

`compile_expression(raw, resolve, where)` returns an `Expression` holding a small
tuple tree: `("num", v)`, `("ref", path)`, `("neg", node)` and
`("op", operator, left, right)`. References are resolved to full paths **at
compile time**, using a resolver bound to the current scope and declaration
position, so evaluation is a dictionary lookup and cannot fail for scope
reasons. `Expression.refs` records what it depends on, which `needed_fields()`
uses to decide what a reader must read even when the output does not name it.

`Expression.as_int(values, what)` evaluates and requires a non-negative whole
number, which is what `shape`, `count`, `record_count`, `record_size` and `seek`
all call.

## Name resolution

`Schema.resolver(scope, limit, where)` returns a closure that:

1. expands aliases (whole name, then leading segment);
2. gathers candidates from the current scope outward, then the root;
3. rejects zero candidates as unknown, and two or more as ambiguous;
4. rejects a target whose `order` is at or beyond `limit`, which is how forward
   and self references are caught;
5. rejects a target that is not a single value.

Because resolution happens while the tree is being built, a field is registered
before its own `shape` is compiled — that is what makes a self-reference
resolvable and therefore rejectable with a clear message.

## The reader interface

Every reader implements:

```python
class Reader:
    def __init__(self, schema: Schema, path: str, needed: set[str]): ...
    def iter_results(self, wanted: int | None) -> Iterator[tuple[int, dict[str, Any]]]: ...
```

`wanted` is `None` for `--index all`. `needed` is the closure computed by
`needed_fields()`: the output fields plus everything their expressions depend
on. Readers use it to skip work — the binary reader advances past an unneeded
array without materialising it, and the HDF5 reader does not touch an unlisted
dataset at all.

Reader-specific notes:

- **`BinReader`** memory-maps the file and walks it once. `_read_fields` handles
  one level and recurses; `_read_list` builds a repeating structure and calls
  `_collect_columns` to expose each member as one value per element. Arrays are
  copied out of the mapping with `.copy()`, so closing the map cannot invalidate
  a value already emitted. The map is closed in a `finally` that tolerates
  `BufferError`, which occurs when an in-flight traceback still holds a view.
  `layout` records `(path, offset, size)` for `--describe`.
- **`Hdf5Reader`** imports `h5py` lazily so the other formats work without it.
  Indices are discovered by turning `{index}` into a numeric pattern and matching
  it against every object path.
- **`CsvReader`** reads the whole file with `csv.reader` using `utf-8-sig`, then
  resolves each field's column positions once before iterating rows.

## The writer interface

```python
class Writer:
    name: str
    def write(self, record: dict[str, Any], context: dict) -> None: ...
    def close(self, stdout) -> int: ...
```

`context` carries the reserved fields for the current record. `close` returns the
number of records or arrays written and is where buffered output is flushed.
`JsonWriter` buffers because JSON needs a closing bracket; `CsvWriter` buffers a
`StringIO`; `BinWriter` streams to files immediately and buffers only when the
destination is standard output. `BinWriter` serves both `bin` and `npy` through
its `_encode` method.

## Run orchestration

`execute()` checks agreement across schemas, resolves the destination, builds one
writer, prints the startup summary, then loops over groups, files and indices.
`--skip-errors` is handled at the file level: a `DataError` or `OutputError`
raised while reading one file is logged and counted, and the run continues.

`main()` maps every exception class to an exit code, catching `BrokenPipeError`
before the general `OSError` handler — it is a subclass, and treating a closed
pipe as a failure would break the streaming use case. A final catch-all reports
any unanticipated exception and exits 1, so no input can produce a traceback.

## Error taxonomy

| Class | Raised when | Exit code |
|---|---|---|
| `SchemaError` | A schema is malformed or self-inconsistent | 2 |
| `DataError` | A file does not match its schema, or cannot be read | 1 |
| `OutputError` | An output file cannot be written | 1 |
| `UsageError` | The arguments cannot be acted on | 3 |
| `DependencyError` | A library for the chosen input format is missing | 4 |
| `StructReadError` | Base class; never raised directly | — |

`SchemaError` is also raised at read time for conditions that are schema defects
rather than data defects — a record consuming zero bytes, a `record_size` of
zero — because the file is not at fault.

## Debugging a run

The tool is built to explain itself; these are the levers, in the order worth
reaching for.

**`--describe`** answers "is the schema pointing where I think it is?" It shows
every field with the offset it was actually read from and the value found
there, including fields the output never uses. A wrong offset in that table is
usually a missing `pad:` or an endianness mistake.

**`--inspect`** answers "what is actually in this file?" without a schema at
all. For HDF5 it prints the object tree with dtypes, shapes and attributes; for
CSV the guessed delimiter and the columns; for anything else the opening bytes
as hex and text, which is where a magic number is found.

**The startup summary**, printed at INFO by default, answers "what did it
assume?" Every default that was applied rather than declared is listed, along
with the fields the output does not need and which will therefore not be read.
A field appearing in that list when it should have been read is a missing entry
in `output.fields`.

**`-v`** adds per-record progress: the offset and size of each record, the
indices discovered in an HDF5 file, and whether a direct seek was used.

**`-vv`** adds per-field tracing: every read with its byte offset and value,
every seek, and every field skipped. On a misparsed binary file this is
normally conclusive — the first field whose traced value looks wrong is where
the layout and the schema diverge.

**`--log-file PATH`** records everything at trace level regardless of console
verbosity, so a run that failed in a pipeline can be examined afterwards
instead of being repeated with more flags.

**Exit codes** separate the kinds of failure without reading the message: 2
means the schema is wrong, 1 means the file did not match it, 3 means the
command line was wrong. A schema error costs nothing to reproduce — `--validate`
finds it without touching any input.

Two invariants worth knowing when reading the code: every failure the tool
anticipates raises one of the exception classes in `core.py`, and every one that
it does not is caught by the last-resort handler in `main`, which reports it and
exits 1. A traceback reaching the terminal is therefore always a bug in the
tool, never a bad input.

## Performance notes

- Binary reads are memory-mapped and arrays are sliced out of the mapping, so
  peak memory tracks one array rather than the file. A 32 MB capture read for
  scalars only peaks under 4 MB; extracting one 512×512 frame peaks under 3 MB.
  Both bounds are asserted by the test suite.
- Without `record_size`, reaching record N is O(N) because records may be
  variable-length. With it, access is O(1).
- Unlisted arrays are skipped rather than read: the binary reader advances the
  offset, and the HDF5 reader never opens the dataset.
- HDF5 slicing is applied to the dataset, so a decimated read transfers only the
  elements requested.
- CSV is the exception: the whole file is parsed into memory before any field is
  extracted.

## Adding a format

An input format touches three files: a key list and a name in `INPUT_FORMATS`
in `core.py`, a builder method on `Schema` in `schema.py` that validates its own
field entries and registers `Field` objects, and a reader class in `readers.py`
implementing `iter_results` plus an entry in the `READERS` table.

An output format touches three files too: a name in `OUTPUT_FORMATS` in
`core.py` — and in `ARRAY_OUTPUTS` if it carries array data, which is what
applies the one-array rule to it — a writer class in `writers.py`, and a branch
in `make_writer`.

Both also need a section here and a worked example under `examples/`, which the
test suite reads and converts to every output format it supports.

## Project structure

```
structread/
├── structread              launcher script
├── structread.py           front end: CLI, orchestration, describe and inspect
├── core.py                 errors, logging, types, constants
├── expressions.py          expression engine and slice parser
├── schema.py               schema model and validation
├── readers.py              one reader per input format
├── writers.py              one writer per output format
├── README.md               this document
├── examples/
│   ├── bin/{simple,moderate,nested,complex}/
│   ├── hdf5/{simple,moderate,complex}/
│   ├── csv/{simple,moderate,complex}/
│   └── merge/
│       each holding an input file, its schemas, and a README
├── tests/
│   └── test_structread.py  the whole test suite
└── tools/
    └── make_examples.py    regenerates every example input file
```

## Development and testing

### Running the tests

```
python3 -m pytest tests -q
python3 -m coverage run --branch --source=core,expressions,schema,readers,writers,structread \
    -m pytest tests -q
python3 -m coverage report -m
```

HDF5 tests skip automatically when `h5py` is absent; everything else runs
without it.

### Coverage

Figures for this release, regenerated at each one:

| Metric | Value |
|---|---|
| Tests | 558, all passing |
| Line coverage | 99.2% (1530 of 1544 statements) |
| Branch coverage | 99.4% (682 of 686 branches) |

The uncovered remainder is `OSError` handling for conditions that cannot be
provoked portably from a test — a file that becomes unreadable between the
existence check and the open, a memory map that fails on a file the system has
already opened — plus the `BufferError` guard that releases a memory map when an
in-flight traceback still references a view onto it, and the standard-output
redirect on a broken pipe, which is exercised end to end by piping into `head`
but not from inside the suite.

### Test design

**The example matrix.** Every example under `examples/` is validated, read, and
converted to JSON, CSV, raw bin and npy, at `--index all` and at a single index.
Conversions the rules forbid are asserted to fail with a usage error rather than
being skipped, so the refusals are tested alongside the successes. Spot checks
then assert real values: raw output is compared byte-for-byte against the
expected bytes, JSON arrays are reshaped with numpy and compared, and CSV output
is parsed and compared row by row.

**The format matrix.** Every input format is combined with every other in a
single run — `-s A f1 f2 -s B f3 -s C f4` — and directed at each output format in
turn. Group ordering, several files per group, `--index` applied across every
group, a failing file in one group with and without `--skip-errors`, and a
disagreement between two groups are each asserted separately.

**Generator-independent assertions.** Most binary tests compare against values
that `tools/make_examples.py` produced. If the generator and the reader shared a
misunderstanding about byte order or padding, those tests would agree with each
other and both be wrong. A separate group checks values against bytes spelled
out literally in the test — every integer width in both byte orders, a float, a
string cut at its NUL, and a layout read with and without a declared pad to show
that padding is never inferred.

**Fuzzing.** The tool takes user-authored schemas and arbitrary files, so the
contract under test is that it always exits with one of its documented codes and
never shows a traceback. Schemas are damaged by a seeded mutator that deletes
keys, replaces values with wrong types and injects unknown keys; binary inputs
are damaged by random byte flips and truncation; CSV inputs are generated as
random strings over a delimiter-heavy alphabet.

**Property-based round-trips.** A seeded generator builds a random field layout,
the payload is packed with the standard library's `struct`, and the tool must
read back exactly what was packed — across both byte orders, every integer
width, arrays, strings, padding and repeated records.

**Memory.** `tracemalloc` bounds two runs over a 32 MB capture, turning the claim
that reads stream rather than buffer into something that fails if it stops being
true.

**Targeted tests** cover the expression parser, name resolution, aliases, every
rule in [Rules enforced at schema load](#rules-enforced-at-schema-load), damaged
input per format, every `--index` position, multi-schema agreement, every
command-line option and exit status, the 0.4 features end to end, and text
encodings as other tools write them.

### What the tests guarantee

Every behaviour this document describes is covered by a test that exercises the
behaviour itself — the option, the schema key, the refusal, the error message —
rather than by reading the document. The example gallery is the largest part of
that: each example is read and converted to every output format it legally
supports, so a documented feature that stopped working would fail the suite
before anyone noticed the document was wrong.

### Static analysis

```
python3 -m ruff check --line-length 100 *.py tools/ tests/
python3 -m mypy --ignore-missing-imports *.py tools/make_examples.py
```

Both are clean.

### Regenerating the examples

```
python3 tools/make_examples.py examples
```

Rewrites every example input file from its generator, byte for byte identical to
what is shipped. Only needed when an example changes.

## Future work

### Schema reuse (`include:`)

Two schemas that read the same file format but emit different output currently
repeat the whole `input` block. `examples/merge/legacy.yaml` duplicates the
layout in `examples/bin/moderate/capture_meta.yaml` line for line. An `include:`
directive pulling a shared `input` block out of another file would remove that
copy, which matters as soon as a site accumulates a handful of real formats,
each read several different ways: today a layout correction has to be applied to
every copy, and nothing detects a copy that was missed.

### Derived and scaled values

Instrument files habitually store raw counts that mean nothing until a scale and
an offset are applied — an ADC reading that is really millivolts, a tick counter
that is really seconds. The expression engine already exists and is used for
shapes and counts; wiring it to emitted values would allow
`{name: seconds, value: $ticks / 1000}` or `{name: mv, scale: 0.001}`, so the
conversion lives in the schema beside the field it belongs to rather than in
whatever script consumes the output. Without it, every downstream consumer
repeats the same arithmetic and eventually one of them gets it wrong.

### Multiple output blocks per schema

A run produces one output, so pulling metadata to CSV and frames to raw bytes
from the same file means reading it twice. For a directory of large captures
that doubles the I/O. Named output blocks, selected with a flag, would let one
pass write both.

### Index ranges, strides and lists

`--index` takes one value or `all`. `--index 0:100:2` and `--index 1,5,9` would
cover "the first hundred frames" and "these three frames" without either reading
everything or invoking the tool once per frame. With `record_size` already
supporting direct access, a range would be cheap to serve.

### Terminator-delimited binary records

Record lengths must be fixed or derivable from a field already read. Formats
that mark the end of a record with a sentinel instead of declaring its length —
common in older telemetry and log formats — cannot be described at all.

### Bit-level field addressing

The smallest addressable unit is a byte, so a packed status word has to be read
as an integer and taken apart afterwards. Declaring a flag's bit position inside
a byte would keep the meaning of each flag in the schema, where the rest of the
layout already is.

### Formatted-text input

Regex-capture and fixed-width layouts would extend the tool to instrument logs
and printed reports, which are the remaining common shape of "a file with a
structure someone can describe".

### Appending rather than replacing

Output files are replaced. Collecting a table incrementally — a nightly run
adding rows to one CSV — needs an append mode that writes the header only when
creating the file.

### Compressed inputs

Reading `.gz` and `.bz2` directly would avoid staging decompressed copies of
archived captures, at the cost of losing direct access to record N in the binary
reader.

### Reading many files in parallel

Files within a run are read one at a time. Thousands of small HDF5 files are
dominated by open and close cost, which a worker pool would hide; output
ordering would have to be preserved to keep runs reproducible.

### HDF5 compound types

Compound and variable-length dataset types are not converted. Flattening a
compound dataset into named fields would mirror what the binary reader already
does for C structures.

## Changelog

### 0.5

Changed:

- The tool is split from one 2200-line module into six, one per layer of the
  pipeline: `core.py`, `expressions.py`, `schema.py`, `readers.py`,
  `writers.py` and `structread.py`. The dependency graph is acyclic and
  one-directional, and `structread.py` re-exports the public names so
  `import structread` still reaches everything.
- Each module opens with a docstring stating what it is responsible for and
  what it may depend on.
- The developer reference gains a [Debugging a run](#debugging-a-run) section
  covering `--describe`, `--inspect`, the startup summary, `-v`, `-vv`,
  `--log-file` and what each exit code narrows the problem down to.

Removed:

- The documentation conformance test module. Checking the README by parsing it
  tested the document rather than the tool; the behaviours it described are
  covered by tests that exercise those behaviours directly.

### 0.4

Fixed:

- A UTF-8 byte-order mark left `\ufeff` inside the first CSV column name, so
  every lookup on that column failed. CSV input is now read as `utf-8-sig`.
- An exception type the tool did not anticipate escaped as a Python traceback.
  A last-resort handler now reports it and exits 1, so no input can produce a
  traceback.
- A record whose final slot was short under `record_size` was dropped even when
  all of its declared fields were present; it is now emitted with a warning.

Added:

- `seek:` moves to an absolute offset, given as an integer or an expression, so
  a header that declares where its data begins can be followed without counting
  the gap by hand.
- `record_size:` declares a fixed record slot. Slack after the declared fields
  is skipped, and record N is reached directly instead of by walking the records
  before it.
- Repeating structures: a structure with a `count:` produces an array of nested
  structures, to any depth. Members are addressable individually, each yielding
  one value per element.
- `expect:` checks that a value holds a declared constant, which is how a magic
  number rejects the wrong file.
- `enum:` decodes a raw value into a name, and rejects a value the table does
  not list.
- `slice:` reads part of an array. On HDF5 the slice is applied to the dataset
  rather than to the result, so a large frame is never read in full.
- `npy` as an output format, carrying dtype, shape and byte order with the data.
- `--describe` shows how a schema resolves against a file: every field's
  location, size and value.
- `--inspect` shows what is inside a file with no schema at all — the HDF5 tree,
  a CSV's guessed dialect and columns, or the opening bytes as hex.
- `examples/bin/nested`, demonstrating repeating structures, `seek:`,
  `record_size:`, `expect:` and `enum:` together.
- `examples/hdf5/moderate/frames_thumb.yaml`, demonstrating lazy slicing and
  `npy` output.

Changed:

- The README is reorganised into a user section, an advanced section and a
  developer reference.
- The key sets the schema validator accepts are module constants, so the
  vocabulary the tool recognises is readable in one place.

Testing:

- Added generator-independent assertions, schema and input fuzzing,
  property-based round-trips, and memory ceilings enforced with `tracemalloc`.
- Added an encoding matrix: BOM, CRLF, missing trailing newline, latin-1, and a
  field beyond the CSV parser's limit.
- 330 tests to 558.

### 0.3

Fixed:

- An alias sharing a name with a field silently overrode that field, because
  aliases are expanded before references are resolved. Such an alias is now
  rejected at schema load; an alias naming the field it points at remains
  allowed.
- A CSV field combining `row:` with `shape:` or `columns:` was accepted and the
  `row:` silently ignored. It is now rejected.

Changed:

- `examples/bin/complex` gained a third level of nesting, an arithmetic shape,
  a shape mixing a literal with an expression, and a record with a zero-length
  dimension.
- `examples/merge` gained `tabular.yaml`, so the merge example covers all three
  input formats rather than two.

Testing:

- Added a format matrix covering every input format combined with every other in
  one run, directed at each output format.
- Added end-to-end tests for every schema construct that was previously only
  validated, and value assertions for the signed integer widths.
- 284 tests to 330.

### 0.2

Ground-up redesign around two concrete tasks: extracting one array to raw bytes
or JSON, and extracting scalar values to CSV or JSON. The schema now describes
the output as well as the input.

Added:

- CSV as an input format; CSV and raw bin as output formats.
- An `output` block in the schema: format, path, and an explicit field list with
  renaming.
- Several schemas in one run via repeated `-s`, letting differently-structured
  files merge into one output.
- Nested structures with lexical scoping, full-path references, and ambiguity
  reported rather than resolved silently.
- `aliases` for short names in paths and expressions.
- Reserved `source`, `stem`, `index` and `schema` fields.
- Default field names from CSV column names and HDF5 attribute or dataset names.
- Optional `version`, `source` and `description` in the schema header.
- Logging through the standard `logging` module, a startup summary of every
  default and assumption applied, `-v`/`-vv`, and `--log-file`.
- `--skip-errors` for batch runs.
- An `examples/` gallery of annotated inputs, verified by the test suite.

Changed:

- `--index` defaults to `all`.
- The index placeholder is `{index}`; `{n}` is available through `aliases`.
- Declared `type` and `shape` validate HDF5 and CSV input rather than
  instructing it.

Removed from 0.1:

- JSON as an input format.
- Sidecar array files and `--outdir`, `--prefix`, `--force`, `--array-mode`,
  `--array-format`; arrays now go to the single chosen output.
- The `summary` and `inline` array modes and `--stats`.
- Marker validation, enum decoding and forward resynchronisation, all of which
  returned in 0.4 in a simpler form.
- Multi-group layouts, `until_eof` groups, `--jsonl`.
- The strict/tolerant distinction; `--skip-errors` is the single opt-out.

### 0.1

Initial release. Binary, HDF5 and JSON input; JSON output with sidecar array
files; marker validation with resynchronisation; multi-group layouts; sidecar,
summary and inline array modes.
