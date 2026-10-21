#!/usr/bin/env python3
# kvsnap K1/K2/K4 end-to-end driver (run.sh): talks the engine line protocol (GEN/KVSYNC) with
# explicit token ids. Phases run1/run2/run3 are separate engine starts (see
# run.sh); state between them lives in logs/kv_e2e_state.json.
import json, os, socket, subprocess, sys

ROOT = os.environ.get('ROOT', os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
PORT = int(os.environ.get('ENGINE_PORT', '8740'))
STATE = ROOT + '/logs/kv_e2e_state.json'
EOS = [248046, 248044]


class Tok:
    def __init__(self):
        self.p = subprocess.Popen([ROOT + '/build/tok_cli', ROOT + '/models/tokenizer'],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True)

    def _q(self, o):
        self.p.stdin.write(json.dumps(o, ensure_ascii=False) + '\n')
        self.p.stdin.flush()
        return json.loads(self.p.stdout.readline())

    def enc(self, s):
        return self._q({'op': 'encode', 'text': s})['ids']

    def dec(self, ids):
        return self._q({'op': 'decode', 'ids': ids, 'skip_special_tokens': False})['text']


def gen(ids, max_tokens, req, snaps=None, ckpt=None, stop_after=None, stop_extra=8):
    """-> (tokens, D-line fields). stop_after: cancel once that token was
    emitted and stop_extra more followed."""
    line = 'GEN %d %d %d %s %d %s' % (req, max_tokens, len(EOS), ' '.join(map(str, EOS)),
                                      len(ids), ' '.join(map(str, ids)))
    if snaps:
        line += ' SNAPS %d %s' % (len(snaps), ' '.join(map(str, snaps)))
    if ckpt:
        line += ' CKPT %d %s' % (len(ckpt), ' '.join(map(str, ckpt)))
    s = socket.create_connection(('127.0.0.1', PORT))
    s.sendall((line + '\n').encode())
    f = s.makefile('r')
    toks, d, seen, sent_x = [], None, -1, False
    for l in f:
        p = l.split()
        if not p:
            continue
        if p[0] == 'T' and int(p[1]) == req:
            toks.append(int(p[2]))
            if stop_after is not None and seen < 0 and toks[-1] == stop_after:
                seen = len(toks)
            if seen >= 0 and not sent_x and len(toks) >= seen + stop_extra:
                s.sendall(b'X %d\n' % req)
                sent_x = True
        elif p[0] == 'D':
            d = p
            break
    s.close()
    return toks, d


def kvsync():
    s = socket.create_connection(('127.0.0.1', PORT))
    s.sendall(b'KVSYNC\n')
    r = s.makefile('r').readline().strip()
    s.close()
    return r


def doc(paths, n, tok):
    """Token ids of the files' text, repeated/truncated to n."""
    txt = ''
    for p in paths:
        with open(ROOT + '/' + p, encoding='utf-8', errors='replace') as fh:
            txt += fh.read() + '\n\n'
    ids = tok.enc(txt)
    while len(ids) < n:
        ids += ids
    return ids[:n]


def chat(tok, sys_ids, user_ids, question):
    """-> ids, cut after the system message, cut before the last <|im_end|>."""
    a = tok.enc('<|im_start|>system\n') + sys_ids + tok.enc('<|im_end|>\n')
    b = tok.enc('<|im_start|>user\n') + user_ids + tok.enc('\n\n' + question)
    tail = len(a) + len(b)
    ids = a + b + tok.enc('<|im_end|>\n<|im_start|>assistant\n<think>\n')
    return ids, len(a), tail


fails = 0


def check(ok, what, info=''):
    global fails
    print('CHECK %s  %s%s' % ('PASS' if ok else 'FAIL', what, ('  ' + info) if info else ''),
          flush=True)
    if not ok:
        fails += 1


def show(tok, name, toks, d):
    t = tok.dec(toks).replace('\n', '\\n')
    print('%s: %d tokens, D=%s\n   %s' % (name, len(toks), ' '.join(d or []),
                                         t[:240] + ('...' if len(t) > 240 else '')), flush=True)


def main():
    phase = sys.argv[1]
    tok = Tok()
    tc, et = tok.enc('<tool_call>'), tok.enc('</think>')
    assert len(tc) == 1 and len(et) == 1, (tc, et)
    tc, et = tc[0], et[0]
    st = json.load(open(STATE)) if os.path.exists(STATE) and phase != 'run1' else {}
    if phase == 'run1':
        S = doc(['docs/CONCURRENCY.md', 'docs/GGUF.md'], 5200, tok)
        A = doc(['README.md', 'docs/QUICKSTART.md', 'docs/BUILD.md'], 8000, tok)
        A1, cut_sys, cut_tail = chat(tok, S, A, '只用一句话回答：上面这份文档介绍的是什么项目？简短思考即可。')
        # R0: first turn; cancelled 8 tokens after </think> (the pause)
        G0, d = gen(A1, 3000, 1, snaps=[cut_sys, cut_tail], ckpt=[tc, et], stop_after=et)
        show(tok, 'R0', G0, d)
        j = G0.index(et) if et in G0 else -1
        check(j >= 0, 'R0 生成里出现 </think>（暂停点）', 'j=%d' % j)
        if j < 0:
            sys.exit(1)
        n_pause = len(A1) + j
        ids_c = A1[:cut_tail] + tok.enc('\n\n（补充：回答时请提到作者。）')
        ids_p = A1 + G0[:j] + [et] + tok.enc('\n\n换一种说法：')
        print('A1 %d tokens, cut_sys %d, cut_tail %d, pause at %d' % (len(A1), cut_sys, cut_tail,
                                                                     n_pause), flush=True)
        # Rc: RAM restore of the tail cut (device-captured checkpoint): control
        C1, d = gen(ids_c, 64, 2)
        show(tok, 'Rc(RAM)', C1, d)
        # Rp: RAM restore of the pause checkpoint
        O1, d = gen(ids_p, 64, 3)
        show(tok, 'Rp(RAM)', O1, d)
        print('KVSYNC', kvsync(), flush=True)
        st = dict(n_pause=n_pause, cut_tail=cut_tail, ids_c=ids_c, ids_p=ids_p, C1=C1, O1=O1)
        json.dump(st, open(STATE, 'w'))
        print('N_PAUSE=%d CUT_TAIL=%d' % (n_pause, cut_tail))
    elif phase == 'run2':
        C2, d = gen(st['ids_c'], 64, 11)
        show(tok, 'Rc(SSD)', C2, d)
        O2, d = gen(st['ids_p'], 64, 12)
        show(tok, 'Rp(SSD)#1', O2, d)
        O3, d = gen(st['ids_p'], 64, 13)
        show(tok, 'Rp(SSD)#2', O3, d)
        print('KVSYNC', kvsync(), flush=True)
        check(C2 == st['C1'], '对照：切点检查点 RAM 恢复 == SSD 恢复（逐 token）')
        check(O2 == st['O1'] and O3 == st['O1'],
              'K1：暂停点检查点 SSD 恢复 == RAM 恢复（逐 token，两次）')
        print('N_PAUSE=%d CUT_TAIL=%d' % (st['n_pause'], st['cut_tail']))
    elif phase == 'run3':
        S2 = doc(['docs/KLD.md', 'docs/PORTING-WINDOWS.md', 'docs/YARN-512K.md'], 5200, tok)
        A2 = doc(['README_EN.md', 'docs/QUICKSTART_EN.md', 'docs/BUILD_EN.md'], 8000, tok)
        Ab, cs, ct = chat(tok, S2, A2, 'Summarize the document above in one sentence.')
        G, d = gen(Ab, 40, 21, snaps=[cs, ct])  # main agent, turn 1
        show(tok, 'main#1', G, d)
        print('KVSYNC', kvsync(), flush=True)
        Bb, _, bt = chat(tok, S2, tok.enc('List three file names mentioned in the system prompt.'),
                         'Answer briefly.')
        Gb, d = gen(Bb, 40, 22, snaps=[cs, bt])  # subagent: same system prompt
        show(tok, 'subagent', Gb, d)
        print('KVSYNC', kvsync(), flush=True)
        A2b = Ab + G + tok.enc('<|im_end|>\n<|im_start|>user\nNow give one more detail.<|im_end|>\n'
                                '<|im_start|>assistant\n<think>\n')
        G2, d = gen(A2b, 64, 23)  # back to the main agent
        show(tok, 'main#2', G2, d)
        check(d is not None and d[2] != 'error' and len(G2) > 0, 'K4：主代理第二轮正常生成')
        print('MAIN1_N=%d SYS_CUT=%d MAIN2_PROMPT=%d' % (len(Ab), cs, len(A2b)))
        if d and len(d) > 10:
            print('MAIN2_CACHED=%s MAIN2_PREFILL_MS=%s' % (d[10], d[5]))
        print('KVSYNC', kvsync(), flush=True)
    print('PHASE %s: %s' % (phase, 'FAIL' if fails else 'PASS'), flush=True)
    sys.exit(1 if fails else 0)


main()
