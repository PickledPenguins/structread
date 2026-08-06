#!/usr/bin/env python3
"""Regenerate every example input file shipped under examples/.

The examples are committed to the project; this script exists so they can be
rebuilt or adjusted without hand-editing binary data.
"""

from __future__ import annotations

import os
import struct
import sys

import numpy as np

# ---------------------------------------------------------------- binary ---


def bin_simple() -> bytes:
    """Header-only device descriptor: fixed scalars and a fixed string."""
    return struct.pack(
        "<I8sHBxf",
        0x44455631,            # magic
        b"SN-4471",            # serial, NUL padded to 8
        0x0102,                # firmware
        1,                     # active
        21.5,                  # temperature
    )


def bin_moderate() -> bytes:
    """Header plus three fixed-size frame records."""
    header = struct.pack("<IHH16s", 0x53545244, 1, 3, b"sample-capture")
    frames = []
    for number in (1, 2, 3):
        pixels = (np.arange(12, dtype=np.int64) + number * 10).astype("<u2")
        frames.append(struct.pack("<QHH2xd", number, 4, 3, 1000.0 + number) + pixels.tobytes())
    return header + b"".join(frames)


def bin_complex() -> bytes:
    """Big-endian nested header and variable-length records read to end of file."""
    header = struct.pack(
        ">I HHB x hh ?? bhiq",
        0xCAFEBABE,            # magic
        640, 480, 3,           # geometry: width, height, planes (+1 pad byte)
        -12, 34,               # geometry.origin: x, y
        True, False,           # flags: calibrated, inverted
        -8, -1600, -320000, -64000000000,   # limits: i8, i16, i32, i64
    )
    records = []
    for number, (rows, cols) in enumerate(((2, 3), (1, 4), (3, 2), (0, 2)), start=1):
        cube = (np.arange(rows * cols * 3, dtype=np.int64) % 200 + number).astype(">u1")
        edges = (np.arange(rows + cols, dtype=np.int64) + 100).astype(">u1")
        pair = (np.arange(2 * cols, dtype=np.int64) + 200).astype(">u1")
        records.append(struct.pack(">IHHd", number, rows, cols, 1.5 * number)
                       + cube.tobytes() + edges.tobytes() + pair.tobytes())
    return header + b"".join(records)


def bin_nested() -> bytes:
    """Header with an absolute data offset, then records holding a point list."""
    data_offset = 48
    record_size = 48
    header = struct.pack(
        "<IHBxHxxI24s",
        0x43415431,            # magic, "CAT1"
        2,                     # version
        1,                     # kind: 1 = image
        3,                     # entry_count
        data_offset,           # where the entries begin
        b"sample catalogue",
    )
    header = header.ljust(data_offset, b"\x00")
    entries = []
    for number, points in enumerate(((2, 1), (1,), (3, 2)), start=1):
        body = struct.pack("<I12sHxx", number, f"entry-{number}".encode(), len(points))
        for position, weight in enumerate(points):
            body += struct.pack("<fff", float(position), float(position) * 2, float(weight))
        entries.append(body.ljust(record_size, b"\x00"))
    return header + b"".join(entries)


# ------------------------------------------------------------------ hdf5 ---


def hdf5_simple(path: str) -> None:
    """Root attributes only."""
    import h5py

    with h5py.File(path, "w") as handle:
        handle.attrs["run_id"] = np.uint32(42)
        handle.attrs["instrument"] = "sample-camera"
        handle.attrs["gain"] = 1.5
        handle.attrs["operator"] = "unattended"


def hdf5_moderate(path: str) -> None:
    """Frame groups with per-frame attributes and image datasets."""
    import h5py

    with h5py.File(path, "w") as handle:
        handle.attrs["run_id"] = np.uint32(42)
        handle.attrs["instrument"] = "sample-camera"
        data = handle.create_group("data")
        for number in (1, 2, 10):
            group = data.create_group(f"frame{number}")
            group.attrs["timestamp"] = 1000.0 + number
            group.attrs["exposure_us"] = np.uint32(250 * number)
            group.attrs["gain"] = 1.5
            group.create_dataset(
                "data", data=(np.arange(24, dtype=np.uint16) + number).reshape(4, 6))


def hdf5_complex(path: str) -> None:
    """Indexed names inside a path segment, sibling shape attributes, scalar dataset."""
    import h5py

    with h5py.File(path, "w") as handle:
        handle.attrs["experiment"] = "scan-A"
        handle.create_dataset("calibration", data=np.float64(0.977))
        scan = handle.create_group("scan")
        for number in (1, 2, 3):
            dataset = scan.create_dataset(
                f"img_{number}_raw",
                data=(np.arange(15, dtype=np.float32) * 0.5 + number).reshape(3, 5))
            dataset.attrs["rows"] = np.uint16(3)
            dataset.attrs["cols"] = np.uint16(5)
            meta = scan.create_group(f"img_{number}_meta")
            meta.attrs["filter"] = "narrowband"
            meta.attrs["exposure_us"] = np.uint32(100 * number)


# ------------------------------------------------------------------- csv ---


CSV_SIMPLE = """\
timestamp,serial_no,ambient,signal
1001.0,SN-4471,21.5,0.021
1002.0,SN-4471,21.6,0.024
1003.0,SN-4471,21.4,0.019
"""

CSV_MODERATE = """\
timestamp,serial_no,ambient,signal,x,y,z
1001.0,SN-4471,21.5,0.021,1.0,2.0,3.0
1002.0,SN-4471,21.6,0.024,1.5,2.5,3.5
1003.0,SN-4471,21.4,0.019,2.0,3.0,4.0
1004.0,SN-4471,,0.031,2.5,3.5,4.5
"""

CSV_COMPLEX = """\
# instrument log, exported 2026-05-01
# columns: stamp;station;note;reading
1001.0;alpha;"routine; nothing to report";21.5
1002.0;beta;"multi-line
observation";21.6
1003.0;gamma;;
1004.0;delta;"quoted ""double"" quotes";22.1
"""


# ----------------------------------------------------------------- driver ---


def write(path: str, payload: bytes | str) -> None:
    """Write one example file, creating its directory."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mode, data = ("wb", payload) if isinstance(payload, bytes) else ("w", payload)
    with open(path, mode) as handle:
        handle.write(data)


def main(argv: list[str]) -> int:
    """Write every example input under the directory named on the command line."""
    root = argv[1] if len(argv) > 1 else "examples"
    write(os.path.join(root, "bin", "simple", "device.bin"), bin_simple())
    write(os.path.join(root, "bin", "moderate", "capture.bin"), bin_moderate())
    write(os.path.join(root, "bin", "complex", "scan.bin"), bin_complex())
    write(os.path.join(root, "bin", "nested", "catalog.bin"), bin_nested())
    write(os.path.join(root, "csv", "simple", "readings.csv"), CSV_SIMPLE)
    write(os.path.join(root, "csv", "moderate", "sensors.csv"), CSV_MODERATE)
    write(os.path.join(root, "csv", "complex", "log.csv"), CSV_COMPLEX)
    try:
        import h5py  # noqa: F401
    except ImportError:
        print("h5py not installed: HDF5 examples skipped", file=sys.stderr)
        return 0
    for name, builder in (("simple/instrument.h5", hdf5_simple),
                          ("moderate/frames.h5", hdf5_moderate),
                          ("complex/scan.h5", hdf5_complex)):
        target = os.path.join(root, "hdf5", name)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        builder(target)
    print(f"examples written to {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
