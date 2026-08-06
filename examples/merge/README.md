# merge

Three schemas, three file formats, one table. `legacy.yaml` reads the binary
capture from `../bin/moderate`, `modern.yaml` reads the HDF5 capture from
`../hdf5/moderate`, and `tabular.yaml` reads the CSV log from `../csv/moderate`.
Their `output` blocks emit the same field names in the same order, so one
command merges all three.

Shows: repeated `-s` groups, agreement between schemas, field renaming as the
mechanism that makes unlike inputs agree, and the reserved `schema` field.

```
$ structread -s legacy.yaml  ../bin/moderate/capture.bin \
             -s modern.yaml  ../hdf5/moderate/frames.h5 \
             -s tabular.yaml ../csv/moderate/sensors.csv
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

The `frame_id` column comes from `record.frame_id` in one schema, from an HDF5
attribute in the second, and from a CSV column in the third. Only the output
names have to match; where the values come from is each schema's own business.

Changing either field list, or either output format, makes the run fail rather
than silently producing a misaligned table:

```
ERROR schema mismatch: 'legacy' emits [schema, stem, index, frame_id,
      timestamp] but 'modern' emits [schema, stem, index, timestamp]; all
      schemas in one run must emit the same fields in the same order
```
