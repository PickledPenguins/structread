# hdf5 / complex

Indexed datasets whose index sits inside a name segment rather than being a
whole path segment, with the image shape taken from attributes on the dataset
itself and metadata held in a parallel group.

Shows: `{index}` inside a name (`img_{n}_raw`), shape resolved from sibling
attributes, a scalar dataset declared with an empty shape, string attributes,
and metadata read from a different group than the data.

```
$ structread -s scan.yaml scan.h5 --index 1
[
  {
    "experiment": "scan-A",
    "calibration": 0.977,
    "rows": 3,
    "cols": 5,
    "filter": "narrowband",
    "exposure": 100,
    "image": {"dtype": "float32", "shape": [3, 5], "data": [1.0, 1.5, ...]}
  }
]
```

`/calibration` is a 0-dimensional dataset. Declaring `shape: []` says so, which
is what makes it usable as a single value — in CSV output, for instance.

The image shape is not hard-coded: `rows` and `cols` are read from the
dataset's own attributes first, then used to check it.

```yaml
- {at: "/scan/img_{n}_raw@rows", name: rows, type: u16}
- {at: "/scan/img_{n}_raw@cols", name: cols, type: u16}
- {at: "/scan/img_{n}_raw", name: image, type: f32, shape: [$rows, $cols]}
```
