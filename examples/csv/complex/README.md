# csv / complex

A CSV exported with a non-default dialect and awkward content.

Shows: no header row, a semicolon delimiter, leading comment lines, positional
columns with explicit names, quoted fields containing the delimiter, an embedded
newline and escaped quotes, and empty cells in the final columns.

```
$ head -4 log.csv
# instrument log, exported 2026-05-01
# columns: stamp;station;note;reading
1001.0;alpha;"routine; nothing to report";21.5
1002.0;beta;"multi-line
```

```
$ structread -s log.yaml log.csv --index 1
[
  {
    "stamp": 1002.0,
    "station": "beta",
    "note": "multi-line\nobservation",
    "reading": 21.6
  }
]
```

With `header: false`, columns are addressed by position and every field needs
an explicit `name:`. The two comment lines are removed by `skip_lines: 2`.

Row index 2 has empty `note` and `reading` cells. Text reads back as `""`;
a numeric field with an empty cell reads back as `null`.
