# hdf5 / moderate

Frame groups under `/data/frameN`, each with its own attributes and image
dataset. Two schemas read the same file: metadata to a table, image data to raw
bytes.

Shows: `{index}` in a path, the `n: index` alias, natural numeric ordering,
declared type and shape used to validate a dataset, lazy slicing, and `npy`
output.

## Metadata to CSV

```
$ structread -s frames_meta.yaml frames.h5
stem,index,run_id,instrument,stamp,exposure
frames,1,42,sample-camera,1001.0,250
frames,2,42,sample-camera,1002.0,500
frames,10,42,sample-camera,1010.0,2500
```

The file contains `frame1`, `frame2` and `frame10`. They are visited in natural
numeric order, so `frame10` comes last, and `index` holds the real frame number
rather than a running counter.

## Image data to raw bytes

```
$ structread -s frames_image.yaml frames.h5 --index 2 -o frame2.bin
$ ls -l frame2.bin
-rw-r--r-- 1 user user 48 ... frame2.bin
```

`frames_image.yaml` declares `type: u16` and `shape: [4, 6]`. A file whose
frames are a different type or size fails immediately instead of writing a
buffer nobody can interpret:

```
ERROR other.h5: /data/frame2/data has shape [8, 6], schema declares [4, 6]
```

## A decimated copy, as npy

`frames_thumb.yaml` takes every other pixel and writes numpy's own format, which
carries dtype, shape and byte order with the data:

```
$ structread -s frames_thumb.yaml frames.h5 --index 1 -o thumb.npy
$ python3 -c "import numpy; print(numpy.load('thumb.npy').shape)"
(2, 3)
```

The slice is applied to the dataset, not to the result, so the full 4x6 frame is
never read. On a real 2048x2048 image that is the difference between reading
8 MB and reading 2 MB.
