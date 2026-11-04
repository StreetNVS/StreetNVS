"""Read TFRecord payloads with length and data CRC32C checks."""

import struct
import google_crc32c


def masked_crc(data):
    c = google_crc32c.value(data)
    return (c >> 15 | c << 17 & 4294967295) + 2726488792 & 4294967295


def records(path):
    with open(path, "rb") as f:
        while True:
            header = f.read(12)
            if not header:
                return
            if len(header) != 12:
                raise ValueError("Truncated TFRecord header")
            size, crc = struct.unpack("<QI", header)
            if crc != masked_crc(header[:8]):
                raise ValueError("TFRecord length CRC mismatch")
            if size > 512 * 1024 * 1024:
                raise ValueError("Unreasonable TFRecord size")
            data = f.read(size)
            checksum = f.read(4)
            if len(data) != size or len(checksum) != 4:
                raise ValueError("Truncated TFRecord data")
            if struct.unpack("<I", checksum)[0] != masked_crc(data):
                raise ValueError("TFRecord data CRC mismatch")
            yield data
