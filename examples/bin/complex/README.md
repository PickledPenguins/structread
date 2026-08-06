# bin / complex

A big-endian file with nested header structures and variable-length records.
Each record carries its own dimensions, so no two records are the same size and
the file is walked to its end rather than using a declared count.

Shows: nesting three levels deep, an alias reaching across scopes, every signed
integer width, `bool`, a 3-D array, an arithmetic shape, a shape mixing a
literal with an expression, a zero-length dimension, big-endian byte order, and
records read until end of file.

## Scope

Inside `record`, `$rows` and `$cols` are siblings and resolve by short name.
`header.geometry.planes` is in a different branch of the tree, so a short name
would not find it; the schema declares an alias for the full path:

```yaml
aliases:
  planes: header.geometry.planes
...
    - {name: cube,  type: u8, shape: [$rows, $cols, $planes]}
    - {name: edges, type: u8, shape: [$rows + $cols]}
    - {name: pair,  type: u8, shape: [2, $cols]}
```

An alias may name the field it points at, as `planes` does here. An alias that
names a *different* field is rejected, since it would otherwise shadow that
field silently.

`header.geometry.origin` sits three levels down; its fields are reached as
`header.geometry.origin.x` and `header.geometry.origin.y`.

## Output

```
$ structread -s scan.yaml scan.bin --index 0
[
  {
    "header.magic": 3405691582,
    "header.geometry.width": 640,
    "header.geometry.height": 480,
    "header.geometry.planes": 3,
    "header.geometry.origin.x": -12,
    "header.geometry.origin.y": 34,
    "header.flags.calibrated": true,
    "header.flags.inverted": false,
    "header.limits.smallest": -8,
    "header.limits.small": -1600,
    "header.limits.medium": -320000,
    "header.limits.large": -64000000000,
    "record.id": 1,
    "record.rows": 2,
    "record.cols": 3,
    "record.gain": 1.5,
    "record.cube":  {"dtype": "uint8", "shape": [2, 3, 3], "data": [1, 2, ...]},
    "record.edges": {"dtype": "uint8", "shape": [5], "data": [100, 101, ...]},
    "record.pair":  {"dtype": "uint8", "shape": [2, 3], "data": [200, 201, ...]}
  }
]
```

The four records produce cube shapes `[2, 3, 3]`, `[1, 4, 3]`, `[3, 2, 3]` and
`[0, 2, 3]`. The last one has no elements: a zero-length dimension is read
normally and reported with its shape, not treated as an error.
