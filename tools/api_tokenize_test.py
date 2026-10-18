#!/usr/bin/env python3
# api_tokenize_test.py — black-box checks for POST /v1/tokenize and
# POST /v1/detokenize. Only the Python standard library is required.
#
# Start the API first (no engine is needed; a tokenizer must be loaded):
#   ./build/qwenox-win.exe --console --host 127.0.0.1 --port 18731 \
#       --engine 127.0.0.1:1
#   python tools/api_tokenize_test.py --base http://127.0.0.1:18731
import argparse
import json
import sys
import urllib.error
import urllib.request


def call(base, path, payload):
    data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    req = urllib.request.Request(base + path, data=data,
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read()
            try:
                return r.status, json.loads(raw)
            except json.JSONDecodeError:
                return r.status, raw.decode(errors="replace")
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw)
        except json.JSONDecodeError:
            return e.code, raw.decode(errors="replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:18731")
    args = ap.parse_args()
    base = args.base.rstrip("/")
    failures = []

    def check(name, ok, detail=""):
        print(("PASS  " if ok else "FAIL  ") + name +
              ("" if ok or not detail else "  (" + str(detail) + ")"))
        if not ok:
            failures.append(name)

    text = "你好，world 🌍"

    st, j = call(base, "/v1/tokenize", {"text": text})
    ids = j.get("tokens") if isinstance(j, dict) else None
    check("tokenize 200", st == 200, j)
    check("count matches tokens", isinstance(ids, list) and
          j.get("count") == len(ids) and len(ids) > 0, j)

    st, j = call(base, "/v1/detokenize", {"tokens": ids})
    check("encode/decode round-trip",
          st == 200 and j.get("text") == text, j)

    st, jo = call(base, "/v1/tokenize", {"text": text, "return_offsets": True})
    offs = jo.get("offsets") if isinstance(jo, dict) else None
    nbytes = len(text.encode("utf-8"))
    contiguous = (isinstance(offs, list) and bool(offs) and
                  offs[0][0] == 0 and offs[-1][1] == nbytes and
                  all(offs[i][1] == offs[i + 1][0]
                      for i in range(len(offs) - 1)))
    check("offsets cover the UTF-8 byte range",
          st == 200 and len(offs or []) == jo.get("count") and contiguous, jo)

    st, jp = call(base, "/v1/tokenize", {"prompt": text})
    check("prompt alias", st == 200 and jp.get("tokens") == ids, jp)
    st, ji = call(base, "/v1/detokenize", {"ids": ids})
    check("ids alias", st == 200 and ji.get("text") == text, ji)

    st, je = call(base, "/v1/tokenize", {"text": ""})
    check("empty text", st == 200 and je.get("tokens") == [] and
          je.get("count") == 0, je)

    st, js = call(base, "/v1/tokenize", {"text": "<|im_end|>"})
    special = js.get("tokens") if isinstance(js, dict) else None
    check("special token", st == 200 and special and
          js.get("count") == len(special), js)
    _, d1 = call(base, "/v1/detokenize", {"tokens": special})
    _, d2 = call(base, "/v1/detokenize",
                 {"tokens": special, "skip_special_tokens": False})
    check("skip_special_tokens=true drops it", d1.get("text") == "", d1)
    check("skip_special_tokens=false keeps it",
          d2.get("text") == "<|im_end|>", d2)

    for name, path, payload in [
        ("missing text", "/v1/tokenize", {}),
        ("text not a string", "/v1/tokenize", {"text": 123}),
        ("missing tokens", "/v1/detokenize", {}),
        ("tokens not an array", "/v1/detokenize", {"tokens": "x"}),
        ("negative id", "/v1/detokenize", {"tokens": [-1]}),
        ("out-of-range id", "/v1/detokenize", {"tokens": [99999999]}),
        ("float id", "/v1/detokenize", {"tokens": [1.5]}),
    ]:
        st, body = call(base, path, payload)
        check(name + " -> 400", st == 400, body)
    st, body = call(base, "/v1/tokenize", b"{")
    check("bad JSON -> 400", st == 400, body)

    try:
        with urllib.request.urlopen(base + "/health", timeout=10) as r:
            eps = json.load(r).get("endpoints", [])
        check("/health lists both endpoints",
              "/v1/tokenize" in eps and "/v1/detokenize" in eps, eps)
    except Exception as e:  # noqa: BLE001 - report and fail the check
        check("/health reachable", False, e)

    print("\n%s (%d failed)" % ("FAILED" if failures else "OK", len(failures)))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
