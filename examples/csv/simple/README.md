# csv / simple

A CSV with a header row, read with almost nothing declared.

Shows: field names defaulted from column names, and the `str` type default.

```
$ structread -s readings.yaml readings.csv --index 0
[
  {
    "timestamp": "1001.0",
    "serial": "SN-4471",
    "ambient": "21.5",
    "signal": "0.021"
  }
]
```

Every value is text, because no field declares a `type:`. That is deliberate —
guessing types from cell contents is unpredictable — and the startup summary
calls it out for each field:

```
INFO  assumed   [readings] field "ambient": type str [default] - declare
                type: f32 or similar for numeric output
```

Compare with `csv/moderate`, which declares types and gets numbers.
