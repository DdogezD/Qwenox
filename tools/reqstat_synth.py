#!/usr/bin/env python3
# reqstat_synth.py — 生成合成 reqstat 记录文件（格式见 src/api/reqstat.h）。
#
# 用法:
#   tools/reqstat_synth.py --dir build/reqstat-synth --start-ms 1700000000000 \
#       --seconds 86400 --rate 1 [--files 1] [--prompt 1000] [--cached 500] [--gen 50]
#
# --seconds 是覆盖的时间长度，--rate 是每秒请求数（整数），总记录数 =
# seconds * rate。时间戳在区间内均匀分布；--files 把记录切成多个链文件
# （前面的命名 reqstat-<seq>.bin，最后一个写 reqstat.bin），用于测试多文件。
import argparse
import datetime as dt
import os
import struct
import sys

import numpy as np

MAGIC = 0x47525153  # "GRQS"
VERSION = 1
HSZ, RSZ = 256, 64
RECORD_DTYPE = np.dtype([
    ("ts", "<u8"), ("seq", "<u8"),
    ("prompt", "<u4"), ("cached", "<u4"), ("gen", "<u4"),
    ("proposed", "<u4"), ("commit", "<u4"), ("rounds", "<u4"),
    ("prefill_us", "<u4"), ("decode_us", "<u4"), ("ttft_us", "<u4"),
    ("flags", "<u4"), ("crc", "<u2"), ("pad", "<u2"), ("reserved", "<u4"),
])
assert RECORD_DTYPE.itemsize == RSZ


def crc16_table():
    tbl = np.zeros(256, dtype=np.uint16)
    for i in range(256):
        c = i << 8
        for _ in range(8):
            c = ((c << 1) ^ 0x1021) & 0xFFFF if c & 0x8000 else (c << 1) & 0xFFFF
        tbl[i] = c
    return tbl


CRC_TABLE = crc16_table()


def crc16_rows(raw):
    """CRC-16/CCITT-FALSE over each row of a (N, 56) uint8 array."""
    crc = np.full(raw.shape[0], 0xFFFF, dtype=np.uint16)
    for i in range(raw.shape[1]):
        crc = CRC_TABLE[((crc >> 8) ^ raw[:, i]).astype(np.uint16)] ^ (crc << 8)
    return crc


def write_header(f, seq, count, first_ts, last_ts, rotate_bytes):
    h = bytearray(HSZ)
    now = int(dt.datetime.now(dt.timezone.utc).timestamp() * 1000)
    struct.pack_into("<IHHIIQQQQQ", h, 0, MAGIC, VERSION, HSZ, RSZ, seq,
                     count, first_ts, last_ts, now, rotate_bytes)
    f.write(h)


def build_chunk(start_idx, count, start_ms, rate, prompt, cached, gen):
    buf = np.zeros(count, dtype=RECORD_DTYPE)
    idx = np.arange(start_idx, start_idx + count, dtype=np.uint64)
    if rate == 1:
        ts = np.uint64(start_ms) + idx * np.uint64(1000)
    else:
        ts = np.uint64(start_ms) + (idx * np.uint64(1000)) // np.uint64(rate)
    buf["ts"] = ts
    buf["seq"] = idx + np.uint64(1)
    buf["prompt"] = prompt
    buf["cached"] = cached
    buf["gen"] = gen
    buf["prefill_us"] = 10000
    buf["decode_us"] = 5000
    buf["ttft_us"] = 2000
    buf["flags"] = 1 | (1 << 4)  # finish=stop, drafter=mtp
    raw = buf.view(np.uint8).reshape(count, RSZ)
    buf["crc"] = crc16_rows(raw[:, :56])
    return raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--start-ms", type=int, required=True,
                    help="first record timestamp (epoch ms, UTC)")
    ap.add_argument("--seconds", type=int, required=True)
    ap.add_argument("--rate", type=int, default=1, help="records per second")
    ap.add_argument("--files", type=int, default=1)
    ap.add_argument("--prompt", type=int, default=1000)
    ap.add_argument("--cached", type=int, default=500)
    ap.add_argument("--gen", type=int, default=50)
    ap.add_argument("--rotate-mb", type=int, default=256)
    args = ap.parse_args()
    if args.rate < 1 or args.files < 1:
        ap.error("--rate and --files must be positive")

    total = args.seconds * args.rate
    os.makedirs(args.dir, exist_ok=True)
    chunk = 200000
    written = 0
    bounds = [round(total * k / args.files) for k in range(args.files + 1)]
    for fi in range(args.files):
        n = bounds[fi + 1] - bounds[fi]
        if n <= 0:
            continue
        seq = fi
        name = "reqstat.bin" if fi == args.files - 1 else "reqstat-%06d.bin" % seq
        path = os.path.join(args.dir, name)
        first_ts = args.start_ms + (bounds[fi] * 1000) // args.rate
        last_ts = args.start_ms + ((bounds[fi + 1] - 1) * 1000) // args.rate
        with open(path, "wb") as f:
            write_header(f, seq, n, first_ts, last_ts,
                         args.rotate_mb * 1048576)
            done = 0
            while done < n:
                take = min(chunk, n - done)
                raw = build_chunk(bounds[fi] + done, take, args.start_ms,
                                  args.rate, args.prompt, args.cached, args.gen)
                f.write(raw.tobytes())
                done += take
                written += take
                print("\r%d/%d" % (written, total), end="", file=sys.stderr,
                      flush=True)
        print("\n%s: %d records, %.1f MiB" %
              (path, n, (HSZ + n * RSZ) / 1048576.0), file=sys.stderr)


if __name__ == "__main__":
    main()
