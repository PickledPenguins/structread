# bin / nested

A catalogue file whose records each hold a variable-length array of nested
structures, reached through an absolute offset carried in the header.

Shows: a repeating nested structure (`count:` on a structure), a structure
inside each element, `seek:` to an absolute offset, `record_size:` for fixed
record slots, `expect:` as a magic-number check, and `enum:` decoding.

## The header ends by seeking

```yaml
    - {name: data_offset, type: u32}
    - {name: description, type: char, shape: [24]}
    - {seek: $data_offset}
```

The reserved space between the header and the first entry never has to be
counted by hand. `pad:` skips a fixed number of bytes; `seek:` jumps to an
absolute offset, which may be an expression over anything already read.

## Entries occupy fixed slots

```yaml
  record_count: $header.entry_count
  record_size: 48
```

Each entry is 48 bytes whether it holds one point or three, so the slack after
the point list is absorbed automatically. A declared `record_size` also means
`--index 2` seeks straight to the third entry instead of walking the first two.

## The point list

```yaml
    - name: points
      count: $npoints
      fields:
        - name: position
          fields:
            - {name: x, type: f32}
            - {name: y, type: f32}
        - {name: weight, type: f32}
```

```
$ structread -s catalog.yaml catalog.bin --index 2 --indent 0
[{"header.magic": 1128354865, "header.version": 2, "header.kind": "image",
  "header.entry_count": 3, "header.data_offset": 48,
  "header.description": "sample catalogue",
  "record.id": 3, "record.label": "entry-3", "record.npoints": 2,
  "record.points": [{"position": {"x": 0.0, "y": 0.0}, "weight": 3.0},
                    {"position": {"x": 1.0, "y": 2.0}, "weight": 2.0}]}]
```

A repeating structure is emitted whole. Its members are also addressable, each
yielding one value per element, which is useful when only part of the list is
wanted:

```yaml
output:
  format: json
  fields:
    - {field: record.id, as: id}
    - {field: record.points.weight, as: weights}
```

```
[{"id": 3, "weights": [3.0, 2.0]}]
```

Because those values are lists, they cannot go to CSV or to raw bytes; JSON is
the destination for anything from a repeating structure.

## Magic and enum

```yaml
    - {name: magic, type: u32, expect: 0x43415431}
    - {name: kind,  type: u8, enum: {1: image, 2: spectrum}}
```

Pointing this schema at the wrong file stops immediately rather than producing
plausible numbers:

```
ERROR other.bin: field header.magic at offset 0: expected 1128354865,
      found 1398035012
```
