#!/usr/bin/env python3
"""codeneedle 离线自测（不需要 GPU）：判分用例 + 假引擎跑通 run/cmp。

  python bench/codeneedle/selftest.py

假引擎讲同样的协议：从 prompt 末尾认出题目里的名字，按设定的正确率作答（同一题同一 seed
用同一个随机数，模拟两轮耦合采样）。跑三轮：A、和 A 完全一样的 same、正确率明显下降的 worse，
要求 cmp(A, same) = PASS、cmp(A, worse) = FAIL。最后一行 SELFTEST PASS|FAIL。
"""
import hashlib
import json
import math
import socket
import subprocess
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

PORT = 18732
FAILS = []


def check(cond, what):
    print(('  ok   ' if cond else '  FAIL ') + what)
    if not cond:
        FAILS.append(what)


def judge_cases():
    qp = dict(kind='path', answer='src/gpu/parts/49_rckpt.inc')
    qn = dict(kind='num', answer='16')
    qf = dict(kind='num', answer='0.05')
    qh = dict(kind='num', answer='0x47525153')
    bn = {'49_rckpt.inc': 1, 'main.cpp': 2}
    cases = [
        (qp, '`src/gpu/parts/49_rckpt.inc`', True),
        (qp, 'src/gpu/parts/49_rckpt.inc。', True),
        (qp, '文件：49_rckpt.inc', True),
        (qp, 'parts/49_rckpt.inc', True),
        (qp, 'src/gpu/parts/40_model.inc', False),
        (dict(kind='path', answer='src/api/main.cpp'), 'main.cpp', False),   # basename 不唯一
        (qp, '不知道', False),
        (qn, '16', True), (qn, '默认值是 16。', True), (qn, '`16`', True), (qn, '0x10', True),
        (qn, '17', False), (qf, '0.05f', True), (qf, '0.050', True), (qh, '0x47525153', True),
        (qh, '1196577107', True), (qn, '', False),
        (qn, 'QWENOX_Q6W_BLOCKS 默认 16', True), (qn, '答案：\n16', True), (qn, 'v2 是 16', True),
        (qn, '1.5', False), (qf, '.05', False), (qp, '在 `src/gpu/parts/49_rckpt.inc` 里', True),
    ]
    for q, out, want in cases:
        got, ext = C.judge(q, out, bn)
        check(got == want, f'judge {q["answer"]!r} <- {out!r}: {got} (want {want})')


class Fake(threading.Thread):
    """mode: (salt, p_correct)。答对与否 = H(salt, 名字, seed) < p。"""

    def __init__(self, answers, tk):
        super().__init__(daemon=True)
        self.answers, self.tk, self.mode = answers, tk, ('A', 0.75)
        self.srv = socket.create_server(('127.0.0.1', PORT))

    def run(self):
        while True:
            c, _ = self.srv.accept()
            threading.Thread(target=self.serve, args=(c,), daemon=True).start()

    def serve(self, c):
        f = c.makefile('rb')
        for raw in f:
            v = raw.decode().split()
            if not v or v[0] != 'GEN':
                continue
            req, n, ne = int(v[1]), int(v[2]), int(v[3])
            i = 4 + ne
            nid = int(v[i])
            ids = list(map(int, v[i + 1:i + 1 + nid]))
            rest = v[i + 1 + nid:]
            seed = rest[rest.index('SAMPLE') + 5] if 'SAMPLE' in rest else 'greedy'
            snap = int(rest[rest.index('SNAPS') + 2]) if 'SNAPS' in rest else 0
            tail = self.tk.decode(ids[-80:])
            name = tail.split('`')[-2] if '只输出答案本身' in tail and tail.count('`') >= 2 else None
            if name is None or name not in self.answers:
                out = 'x'      # 预热请求
            else:
                kind, ans = self.answers[name]
                salt, p = self.mode
                u = int.from_bytes(hashlib.sha256(f'{name}|{seed}'.encode()).digest()[:4], 'little') / 2 ** 32
                ok = u < p
                if kind == 'num':
                    out = ans if ok else str(int(float(C.parse_num(ans))) + 1)
                else:
                    out = ans if ok else ('src/api/main.cpp' if ans != 'src/api/main.cpp' else 'src/api/http.cpp')
            toks = self.tk.encode(out)[:n]
            lp = math.log(0.9 if out != 'x' else 0.5)
            lines = [f'T {req} {t} {lp:.6f}' for t in toks] + [f'T {req} {C.IM_END} -0.01']
            cached = snap if 0 < snap < len(ids) else 0
            lines.append(f'D {req} stop {len(ids)} {len(toks)} 1.0 1.0 0 0 0 {cached} 0')
            c.sendall(('\n'.join(lines) + '\n').encode())


def main():
    print('1. judge')
    judge_cases()
    print('2. fake engine: run A / same / worse, then cmp')
    S = json.load(open(C.HERE / 'set.json', encoding='utf-8'))
    answers = {q['name']: (q['kind'], q['answer']) for q in S['questions']}
    tk = C.tokenizer()
    fake = Fake(answers, tk)
    fake.start()
    run = [sys.executable, str(C.HERE / 'run.py')]
    for tag, mode in (('st_A', ('A', 0.75)), ('st_same', ('A', 0.75)), ('st_worse', ('A', 0.40))):
        fake.mode = mode
        r = subprocess.run(run + ['run', '--tag', tag, '--port', str(PORT), '--only', 'L16K', '--seeds', '2'],
                           capture_output=True, text=True, encoding='utf-8')
        print('   ' + (r.stdout.strip().split('\n')[-1] if r.stdout.strip() else r.stderr[-400:]))
        check(r.returncode == 0, f'run {tag} exit 0')
    d = json.loads((C.ROOT / 'logs/codeneedle/st_A.json').read_text(encoding='utf-8'))
    acc = sum(x['greedy']['ok'] for x in d['res'].values()) / len(d['res'])
    check(0.55 < acc < 0.95, f'fake A greedy accuracy {acc:.2f} ~ 0.75 (judge sees fake answers)')
    check(d['stats']['q_cached_hits'] == d['stats']['q_total'], 'every question request reports a cache hit')
    for b, want in (('st_same', 0), ('st_worse', 1)):
        r = subprocess.run(run + ['cmp', '--a', 'st_A', '--b', b], capture_output=True, text=True, encoding='utf-8')
        print('\n'.join('   ' + ln for ln in r.stdout.strip().split('\n')))
        check(r.returncode == want, f'cmp st_A vs {b}: exit {r.returncode} (want {want})')
    print('SELFTEST ' + ('PASS' if not FAILS else 'FAIL'))
    sys.exit(1 if FAILS else 0)


if __name__ == '__main__':
    main()
