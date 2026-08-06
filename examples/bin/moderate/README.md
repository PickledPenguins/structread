# bin / moderate

A header followed by a counted series of frame records. Two schemas read the
same file: one produces a table of metadata, the other produces raw pixels.

Shows: `record_count` taken from a header field, a shape sized from record
fields, output field renaming, and the two output formats a binary file usually
needs.

## Metadata to CSV

```
$ structread -s capture_meta.yaml capture.bin
stem,index,label,frame_id,width,height,timestamp
capture,0,sample-capture,1,4,3,1001.0
capture,1,sample-capture,2,4,3,1002.0
capture,2,sample-capture,3,4,3,1003.0
```

`record.data` is not in `output.fields`, so the pixel data is never read. The
startup summary says so:

```
INFO  assumed   [capture_meta] not in output, will not be read: header.magic, ...
```

## Pixels to raw bytes

```
$ structread -s capture_frame.yaml capture.bin --index 1 | od -A d -t u2
0000000    20    21    22    23    24    25    26    27
0000016    28    29    30    31
```

Or one file per frame:

```
$ structread -s capture_frame.yaml capture.bin -o "frame_{index}.bin"
$ ls
frame_0.bin  frame_1.bin  frame_2.bin
```
