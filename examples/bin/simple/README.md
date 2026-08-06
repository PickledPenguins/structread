# bin / simple

A header-only binary file: one fixed set of scalars, read once. There is no
`record` block, so the file produces exactly one result no matter what `--index`
says.

Shows: fixed scalars, a `char` string with `shape`, `pad`, `bool`, and the
default output (JSON to standard output, keyed by full field path).

```
$ structread -s device.yaml device.bin
[
  {
    "header.magic": 1145394737,
    "header.serial": "SN-4471",
    "header.firmware": 258,
    "header.active": true,
    "header.temperature": 21.5
  }
]
```

The serial field is 8 bytes on disk holding `SN-4471\0`; `char` cuts the string
at the first NUL.
