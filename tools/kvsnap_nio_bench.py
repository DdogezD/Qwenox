#!/usr/bin/env python3
"""kvsnap 普通 IO vs 直通 IO 对比的客户端（tools/kvsnap_nio_bench.sh 驱动，引擎在 127.0.0.1:8732）。

  save    --state F --ntok N   P = oracle 65536 前 N 个 token：GEN P 16 → 回合结束存盘；
                               再 live 续写 Q = P+out+t2 32 个 token 作为 gold
  restore --state F --reps K   新引擎（RCKPT=0）：GEN Q 走 SSD 恢复，K 次（中间插一个短 prompt
                               把 live 状态换掉），每次输出必须 == gold
  mem     --dir D              系统文件缓存快照（Linux /proc/meminfo + fincore；Windows CIM 内存计数）
  summary --logs L --modes M   解析引擎日志与 mem 文件，打印对比表；两种 IO 的 gold 必须相同
  cmpdir  A B                  两个缓存目录的 c_/m_/k_ 文件逐字节相同
"""
import argparse, hashlib, json, os, re, shutil, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwentok import Tokenizer
from conc_verify import Conn

ROOT = Path(__file__).resolve().parent.parent
WIN = os.name == 'nt'


def sync():
    c = Conn()
    v = c.cmd('KVSYNC')
    c.close()
    return dict(ckpts=int(v[1]), pages=int(v[2]), mtps=int(v[3]), bytes=int(v[4]))


def cmd_save(a):
    toks = [int(x) for x in (ROOT / 'data/qsa-oracle/65536.tokens').read_text().split()]
    P = toks[:a.ntok]
    t2 = Tokenizer().encode('\n\nSummarize the text above in one sentence.\n')
    c = Conn()
    t0 = time.time()
    r1 = c.gen(P, 16, 1)
    w1 = time.time() - t0
    s1 = sync()
    Q = P + r1['tokens'] + t2
    t0 = time.time()
    g = c.gen(Q, 32, 1)
    w2 = time.time() - t0
    c.close()
    s2 = sync()
    print(f'  save: |P|={len(P)} prefill+16 {w1:.1f}s; gold |Q|={len(Q)} n={len(g["tokens"])} '
          f'{w2:.1f}s stats={g["stats"]}', flush=True)
    print(f'  sync after P: {s1}; after gold: {s2}', flush=True)
    Path(a.state).write_text(json.dumps(dict(Q=Q, gold=g['tokens'], sync=s1)))
    return 0 if s1['ckpts'] >= 1 and s1['pages'] >= len(P) // 256 - 1 else 1


def cmd_restore(a):
    st = json.loads(Path(a.state).read_text())
    Q, gold = st['Q'], st['gold']
    X = Tokenizer().encode('Hello, who are you?')
    bad = 0
    for i in range(a.reps):
        c = Conn()
        t0 = time.time()
        r = c.gen(Q, 32, 1)
        w = time.time() - t0
        ok = r['tokens'] == gold
        bad += not ok
        print(f'  restore #{i + 1}: {w:.2f}s (恢复 + 后缀 prefill + 32 token) '
              f'{"== gold" if ok else "!= gold"} stats={r["stats"]}', flush=True)
        if i + 1 < a.reps:
            c.gen(X, 4, 0)  # 换掉 live 状态（短于 KVSNAP_MIN，不存盘）
        c.close()
        sync()
    return 1 if bad else 0


def ps(cmd):
    return subprocess.run(['powershell', '-NoProfile', '-Command', cmd], capture_output=True,
                          text=True, timeout=60).stdout.strip()


def mem_snapshot(d):
    out = {}
    if WIN:
        v = ps('$m=Get-CimInstance Win32_PerfRawData_PerfOS_Memory; '
               '$o=Get-CimInstance Win32_OperatingSystem; '
               '"$($m.StandbyCacheCoreBytes) $($m.StandbyCacheNormalPriorityBytes) '
               '$($m.StandbyCacheReserveBytes) $($m.ModifiedPageListBytes) $($o.FreePhysicalMemory)"')
        try:
            core, norm, res, mod, free_kb = (int(x) for x in v.split())
            out = dict(standby_gb=(core + norm + res) / 2**30, modified_gb=mod / 2**30,
                       free_gb=free_kb / 2**20)
        except ValueError:
            out = dict(error=v[:200])
    else:
        mi = {}
        for ln in Path('/proc/meminfo').read_text().splitlines():
            k, v = ln.split(':', 1)
            mi[k] = int(v.split()[0])
        out = dict(cached_gb=mi['Cached'] / 2**20, free_gb=mi['MemFree'] / 2**20)
        files = [str(p) for p in Path(d).glob('*') if p.suffix in ('.kvc', '.kvm', '.kck')]
        if files and shutil.which('fincore'):
            tot = 0
            for i in range(0, len(files), 200):
                r = subprocess.run(['fincore', '--bytes', '--noheadings', '--output', 'RES',
                                    *files[i:i + 200]], capture_output=True, text=True)
                tot += sum(int(x) for x in r.stdout.split() if x.isdigit())
            out['kvsnap_in_cache_gb'] = tot / 2**30
    return out


def cmd_mem(a):
    print(json.dumps(mem_snapshot(a.dir)))
    return 0


SPLIT = re.compile(r'[\r\n]+')
RE_SAVED = re.compile(r'kvsnap: saved (\d+) tokens .*?\+ (\d+) new pages[^;]*?([\d.]+) MB; '
                      r'capture ([\d.]+)s, (\w+) IO pages write ([\d.]+)s, write ([\d.]+)s')
RE_REST = re.compile(r'kvsnap: restored (\d+) tokens from \S+ in ([\d.]+)s \((\d+) pages, '
                     r'(\d+) MB, (\w+) IO: ckpt ([\d.]+)s, pages read ([\d.]+)s \+ verify '
                     r'([\d.]+)s, io wait ([\d.]+)s, device ([\d.]+)s\)')


def lines(p):
    return SPLIT.split(Path(p).read_text(errors='replace')) if Path(p).exists() else []


def cmd_summary(a):
    L = Path(a.logs)
    bad = 0
    golds = {}
    print('\n---- 保存（第一轮 P 的回合结束：新页写盘）----')
    print(f'{"IO":9s} {"tokens":>7s} {"新页":>5s} {"MB":>7s} {"capture s":>9s} {"页写盘 s":>9s} '
          f'{"MB/s":>7s} {"ckpt 写 s":>9s}')
    for m in a.modes.split():
        sv = [RE_SAVED.search(x) for x in lines(L / f'save-{m}.log')]
        sv = [x for x in sv if x]
        if not sv:
            print(f'{m:9s} 没有 saved 行'); bad += 1; continue
        x = max(sv, key=lambda s: int(s.group(2)))
        mb, pw = float(x.group(3)), float(x.group(6))
        print(f'{m:9s} {x.group(1):>7s} {x.group(2):>5s} {mb:7.0f} {float(x.group(4)):9.2f} '
              f'{pw:9.2f} {mb / pw if pw else 0:7.0f} {float(x.group(7)):9.2f}  [{x.group(5)} IO]')
        if (m == 'direct') != (x.group(5) == 'direct'):
            print(f'  FAIL: {m} 模式实际是 {x.group(5)} IO'); bad += 1
        st = L / f'{m}.json'
        if st.exists():
            golds[m] = json.loads(st.read_text())['gold']
    print('\n---- 恢复（新引擎，SSD → 设备；读盘与校验在预取线程上，与 H2D 重叠）----')
    print(f'{"IO":9s} {"#":>2s} {"总 s":>6s} {"MB":>6s} {"GB/s":>5s} {"ckpt s":>6s} {"页读 s":>6s} '
          f'{"校验 s":>6s} {"等IO s":>6s} {"设备 s":>6s}')
    for m in a.modes.split():
        rs = [RE_REST.search(x) for x in lines(L / f'restore-{m}.log')]
        rs = [x for x in rs if x]
        if not rs:
            print(f'{m:9s} 没有 restored 行'); bad += 1; continue
        for i, x in enumerate(rs):
            s, mb = float(x.group(2)), float(x.group(4))
            print(f'{m:9s} {i + 1:2d} {s:6.2f} {mb:6.0f} {mb / 1024 / s if s else 0:5.2f} '
                  f'{float(x.group(6)):6.2f} {float(x.group(7)):6.2f} {float(x.group(8)):6.2f} '
                  f'{float(x.group(9)):6.2f} {float(x.group(10)):6.2f}  [{x.group(5)} IO]')
            if (m == 'direct') != (x.group(5) == 'direct'):
                print(f'  FAIL: {m} 模式实际是 {x.group(5)} IO'); bad += 1
        if any('direct IO not supported' in x for x in lines(L / f'restore-{m}.log') +
               lines(L / f'save-{m}.log')):
            print(f'  注意：{m} 引擎报告直通 IO 不支持，已退回普通 IO')
    print('\n---- 系统文件缓存（GB）：存盘前 / 存盘后 / 恢复后 ----')
    for m in a.modes.split():
        snaps = []
        for ph in ('before', 'saved', 'restored'):
            p = L / f'mem_{m}_{ph}.json'
            snaps.append(json.loads(p.read_text()) if p.exists() else {})
        keys = sorted({k for s in snaps for k in s if k != 'error'})
        cells = [f'{k} ' + ' / '.join(f'{s[k]:.2f}' if k in s else '-' for s in snaps)
                 for k in keys]
        print(f'{m:9s} ' + '; '.join(cells))
    if len(golds) == 2:
        same = len(set(json.dumps(g) for g in golds.values())) == 1
        print(f'\n两种 IO 的 live gold {"相同" if same else "不同"}')
        bad += not same
    return 1 if bad else 0


def digest(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


def cmd_cmpdir(a):
    def files(d):
        return {p.name: p for p in Path(d).iterdir() if p.suffix in ('.kvc', '.kvm', '.kck')}
    A, B = files(a.a), files(a.b)
    common = sorted(set(A) & set(B))
    diff = [n for n in common if digest(A[n]) != digest(B[n])]
    print(f'  缓存文件：{len(A)} vs {len(B)} 个，同名 {len(common)} 个，内容不同 {len(diff)} 个'
          + (f'（如 {diff[0]}）' if diff else ''))
    return 1 if diff or not common else 0


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)
    p = sp.add_parser('save'); p.add_argument('--state', required=True)
    p.add_argument('--ntok', type=int, default=65536)
    p = sp.add_parser('restore'); p.add_argument('--state', required=True)
    p.add_argument('--reps', type=int, default=3)
    p = sp.add_parser('mem'); p.add_argument('--dir', required=True)
    p = sp.add_parser('summary'); p.add_argument('--logs', required=True)
    p.add_argument('--modes', required=True)
    p = sp.add_parser('cmpdir'); p.add_argument('a'); p.add_argument('b')
    a = ap.parse_args()
    return dict(save=cmd_save, restore=cmd_restore, mem=cmd_mem, summary=cmd_summary,
                cmpdir=cmd_cmpdir)[a.cmd](a)


if __name__ == '__main__':
    sys.exit(main())
