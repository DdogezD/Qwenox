#!/usr/bin/env python3
"""Decode the API token cache (data/tcache.bin) back to text.

Each entry is what the engine saw for one request: prompt ids followed by the
generated ids. Short entries (title / summary side requests) are printed in
full with the reply after the last "<|im_start|>assistant" marked; long ones
get a one-line summary. Standard library only.

  python3 tools/tcache_dump.py logs/check/tcache.bin [--max-len 2000]
"""
import argparse
import json
import os
import struct
import sys


def bytes_to_unicode():
    bs = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) +
          list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def load_vocab(path):
    with open(path, encoding="utf-8") as f:
        t = json.load(f)
    vocab = {i: s for s, i in t["model"]["vocab"].items()}
    added = {a["id"]: a["content"] for a in t.get("added_tokens", [])}
    return vocab, added


def decode(ids, vocab, added, byte_of):
    out, buf = [], bytearray()
    for i in ids:
        if i in added:
            out.append(buf.decode("utf-8", "replace"))
            buf.clear()
            out.append(added[i])
        else:
            buf.extend(byte_of.get(ch, ord("?")) for ch in vocab.get(i, "?"))
    out.append(buf.decode("utf-8", "replace"))
    return "".join(out)


def read_entries(path):
    with open(path, "rb") as f:
        data = f.read()
    if data[:8] != b"GDTC1\0\0\0":
        sys.exit(f"{path}: not a token cache file (magic {data[:8]!r})")
    ver, vocab_n, count = struct.unpack_from("<IIQ", data, 8)
    off, entries = 24, []
    for _ in range(count):
        (n,) = struct.unpack_from("<Q", data, off)
        off += 8
        entries.append(list(struct.unpack_from(f"<{n}i", data, off)))
        off += 4 * n
    return ver, vocab_n, entries


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--tokenizer", default=os.path.join(here, "..", "data", "tokenizer",
                                                        "tokenizer.json"))
    ap.add_argument("--max-len", type=int, default=2000,
                    help="entries up to this many ids are printed in full")
    a = ap.parse_args()
    vocab, added = load_vocab(a.tokenizer)
    byte_of = bytes_to_unicode()
    asst = next((i for i, s in added.items() if s == "<|im_start|>"), None)
    ver, vocab_n, entries = read_entries(a.file)
    print(f"{a.file}: version {ver}, vocab {vocab_n}, {len(entries)} entries")
    for k, ids in enumerate(entries):
        if len(ids) > a.max_len:
            print(f"--- #{k}: {len(ids)} ids (long, skipped)")
            continue
        starts = [j for j, t in enumerate(ids) if t == asst]
        cut = starts[-1] if starts else len(ids)
        print(f"=== #{k}: {len(ids)} ids, reply from id {cut}")
        print(decode(ids[:cut], vocab, added, byte_of))
        print(">>> reply >>>")
        print(decode(ids[cut:], vocab, added, byte_of))
        print("<<< end <<<")


if __name__ == "__main__":
    main()
