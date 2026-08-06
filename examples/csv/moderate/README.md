# csv / moderate

The same shape of file, read two ways: one output row per data row, and whole
columns as arrays.

Shows: renamed columns, declared types, a row-pinned scalar, a single-column
array, a multi-column 2-D array, and an empty cell.

## One row per data row

```
$ structread -s sensors_rows.yaml sensors.csv
stem,index,serial,stamp,ambient
sensors,0,SN-4471,1001.0,21.5
sensors,1,SN-4471,1002.0,21.6
sensors,2,SN-4471,1003.0,21.4
sensors,3,SN-4471,1004.0,
```

`serial` is pinned with `row: 0`, so it is read once and repeats rather than
varying with the index. The last row has an empty `ambient` cell, which becomes
an empty CSV cell — or `null` under `--format json`.

## Whole columns as arrays

```
$ structread -s sensors_trace.yaml sensors.csv
[
  {
    "serial": "SN-4471",
    "signal": {"dtype": "float64", "shape": [4],
               "data": [0.021, 0.024, 0.019, 0.031]},
    "xyz": {"dtype": "float32", "shape": [4, 3],
            "data": [1.0, 2.0, 3.0, 1.5, 2.5, 3.5, 2.0, 3.0, 4.0, 2.5, 3.5, 4.5]}
  }
]
```

No field varies with the index here — `serial` is pinned and the other two read
whole columns — so the file yields one result rather than four.

`shape: [$nrows]` on `signal` is checked against the number of data rows, which
catches a file that is not the length it should be.
