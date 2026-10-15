#!/usr/bin/env python3
"""a4_conc.py -- conc_verify flows plus two multi-chunk long prompts, for
QWENOX_MOE_A4 under concurrent prefill (tools/a4_verify_win.sh drives it).

  seq  --tag T             flows one after another on one connection
  conc --tag T --slots N   N client threads (long prompts start first, so two
                           multi-chunk prefills interleave and the short flows
                           arrive while they run -> D1b lending)
Results go to logs/conc/T.json (same format as conc_verify.py).
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import conc_verify as cv
from a5_ab import prompts

LONG = 20000  # > 3 chunks at the Windows default PREFILL_CHUNK 6144


def flows(tk):
    o8, o32, *_ = prompts(tk)
    tag = lambda s: tk.encode(f'[a4 {s}]\n')
    base = _orig_flows(tk)
    long_flows = [
        [('L20k-a', tag('L20k-a') + o32[:LONG], 48, 0, '')],
        [('L20k-b', tag('L20k-b') + o32[8000:8000 + LONG], 48, 4, '')],
    ]
    # cmd_run(concurrent) moves F[-1] to the front: keep the conversation last
    return long_flows + base[:-1] + [base[-1]]


_orig_flows = cv.flows
cv.flows = flows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['seq', 'conc'])
    ap.add_argument('--tag', required=True)
    ap.add_argument('--slots', type=int, default=2)
    a = ap.parse_args()
    ok = cv.cmd_run(a.tag, a.cmd == 'conc', a.slots)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
