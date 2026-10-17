#!/usr/bin/env python3
"""Fork checkpoint check: new agent sessions whose system prompt ends with a
fresh git commit hash.

Three "sessions" share a long system prompt (~7K tokens, unique salt per run)
that differs only in the commit hash near its end, like an agent client that
snapshots `git log` into its system prompt at session start:

  A  hash 1: nothing cached yet, full prefill.
  B  hash 2: diverges from A at the hash; no checkpoint below it exists, so
     B prefills in full -- and the engine checkpoints the fork point
     (engine log: "fork: req N shares a prefix ... checkpointing at X").
  C  hash 3: diverges at the same place; must resume from B's fork
     checkpoint instead of prefilling from 0.

Without the fork checkpoint C prefills in full like B. Run against a live API
after rebuilding the engine:  python3 tools/fork_ckpt_check.py [--port 8731]
Prints PASS/FAIL; exit code 0 only on PASS.
"""
import argparse
import json
import secrets
import sys
import time
import urllib.request


def post(a, body, timeout=600):
    req = urllib.request.Request(
        f"http://{a.host}:{a.port}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    out["_secs"] = time.time() - t0
    return out


def usage(out):
    u = out.get("usage", {})
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    return u.get("prompt_tokens", 0), cached


def system_prompt(salt, commit):
    lines = [f"Session profile {salt}. You are a careful coding agent."]
    for i in range(1, 231):
        lines.append(f"Guideline {i:04d}: when editing module_{i:04d}.cpp keep the public "
                     f"API stable, update the header comment, and run the unit tests of "
                     f"subsystem {i % 17}.")
    lines += ["", "gitStatus: snapshot taken at the start of the conversation.",
              "Current branch: main", "Status: (clean)", "", "Recent commits:",
              f"{commit} ple: optional lossless bf16 n-gram table",
              "c08657d Merge branch 'main'",
              "28d41b7 api: fix GCC dangling-reference false positive",
              "", "Answer in one short sentence."]
    return "\n".join(lines)


def session(a, salt, tag):
    commit = secrets.token_hex(4)[:7]
    msgs = [{"role": "system", "content": system_prompt(salt, commit)},
            {"role": "user", "content": "Say hello."}]
    out = post(a, {"messages": msgs, "max_tokens": 1, "temperature": 0,
                   "enable_thinking": False})
    p, k = usage(out)
    print(f"[{tag}] commit {commit}: prompt {p}, cached {k}, {out['_secs']:.1f}s", flush=True)
    return p, k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8731, help="qwenox-api port")
    a = ap.parse_args()
    salt = secrets.token_hex(8)
    print(f"salt {salt}: three sessions, system prompts differ only in the commit hash",
          flush=True)
    pa, ka = session(a, salt, "A")
    pb, kb = session(a, salt, "B")
    pc, kc = session(a, salt, "C")
    if pa < 6000:
        print(f"FAIL setup: prompt only {pa} tokens (need > 6000 for a checkpoint)")
        sys.exit(1)
    ok = True
    # B diverges from A at the hash and nothing below it is cached.
    if kb > pb // 10:
        print(f"FAIL setup: B already cached {kb}/{pb} (a checkpoint below the hash "
              "existed; rerun gives a new salt)")
        ok = False
    # C resumes from the fork checkpoint B left ~1 page before the hash.
    want = int(pc * 0.85)
    if kc >= want:
        print(f"PASS fork-checkpoint: C cached {kc}/{pc} >= {want} (B {kb}/{pb})")
    else:
        print(f"FAIL fork-checkpoint: C cached {kc}/{pc} < {want}; engine log should "
              "show 'fork: req ... checkpointing at' for B")
        ok = False
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
