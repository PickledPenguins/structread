# hdf5 / simple

Root attributes only. No field declares a `name:`, so each one is named after
the attribute it reads.

Shows: attribute access with `@`, default field names, and a schema with no
index-dependent fields (one result per file, whatever `--index` says).

```
$ structread -s instrument.yaml instrument.h5
[
  {
    "run_id": 42,
    "instrument": "sample-camera",
    "gain": 1.5,
    "operator": "unattended"
  }
]
```

`operator` declares no `type:`, so the file's own type is used. The other three
declare one, which makes a file with unexpected types fail rather than convert
silently.
