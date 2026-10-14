#!/usr/bin/env python3
"""Request phases for tools/vram_conc_win.sh against the engine at 127.0.0.1:8732.

  python tools/vram_conc.py <dir with 8192.tokens and 131072.tokens>   (SHORT=1: ready + one 8K only)

Prints timestamped phase markers and the engine MEM counters after each phase."""
import os, socket, sys, threading, time
SHORT = os.environ.get('SHORT') == '1'
D = sys.argv[1]
T131 = open(D + '/131072.tokens').read().split()
T8 = open(D + '/8192.tokens').read().split()
G = 1 << 30
def ts(): return time.strftime('%H:%M:%S')
def mem(tag):
    try:
        s = socket.create_connection(('127.0.0.1', 8732), timeout=60)
        f = s.makefile('rb'); s.sendall(b'MEM\n'); v = f.readline().decode().split(); s.close()
        x = [int(a) for a in v[2:]]
        print('%s MEM %-10s device %.2f (peak %.2f) registered %.2f pinned %.2f (peak %.2f) committed %.2f hip_free %.2f/%.2f GiB'
              % (ts(), tag, x[0]/G, x[1]/G, x[2]/G, x[3]/G, x[4]/G, x[5]/G, x[6]/G, x[7]/G), flush=True)
    except Exception as e:
        print(ts(), 'MEM', tag, 'ERR', e, flush=True)
res = {}
def gen(name, ids, ng):
    t0 = time.time()
    try:
        s = socket.create_connection(('127.0.0.1', 8732), timeout=3600)
        f = s.makefile('rb')
        s.sendall(('GEN 1 %d 0 %d %s 4\n' % (ng, len(ids), ' '.join(ids))).encode())
        n, tf = 0, None
        while True:
            v = f.readline().decode().split()
            if not v: res[name] = 'FAIL disconnected'; return
            if v[0] == 'T':
                n += 1; tf = tf or time.time()
            elif v[0] == 'D':
                ok = not (len(v) > 2 and v[2] in ('error', 'cancel'))
                res[name] = '%s prompt %d gen %d ttft %.1f s total %.1f s | %s' % (
                    'OK' if ok else 'FAIL', len(ids), n, (tf or time.time()) - t0, time.time() - t0, ' '.join(v)[:80])
                return
    except Exception as e:
        res[name] = 'FAIL %s' % e
def phase(tag, jobs):
    print('%s BEGIN %s' % (ts(), tag), flush=True)
    th = [threading.Thread(target=gen, args=j) for j in jobs]
    for t in th: t.start(); time.sleep(0.2)
    for t in th: t.join()
    for j in jobs: print('   ', j[0], res.get(j[0]), flush=True)
    print('%s END %s' % (ts(), tag), flush=True)
    time.sleep(3); mem(tag)
mem('ready')
phase('S1_8K', [('s1', T8[:8192], 128)])
if not SHORT: phase('C2_32K', [('a', T131[:32768], 256), ('b', T131[65536:98304], 256)])
rev = T131[::-1]
if not SHORT: phase('C2_100K', [('a', T131[:100000], 128), ('b', rev[:100000], 128)])
time.sleep(10); mem('idle_end')
print(ts(), 'CLIENT DONE', flush=True)
