#!/usr/bin/env python3
"""codeneedle 出题：用引擎自身源码拼长文，自动出答案唯一、可核对的题。

  python3 bench/codeneedle/build.py [--rev REV] [--lengths 16384,32768,65536]
         [--hays 3] [--per-type 4] [--out bench/codeneedle/set.json]

四类题（每篇长文每类 per-type 道，目标位置按深度均匀分布）：
  env_file   环境变量 QWENOX_X 在哪个文件读取            → 文件路径
  func_file  函数 f 定义在哪个文件                         → 文件路径
  const_val  常量 C 的值                                    → 数值
  env_def    环境变量 QWENOX_X 未设置时的默认值             → 数值（要读懂 getenv 后的三元表达式）
筛题规则：答案在全库唯一（只有一处读取/定义）；题目里的名字在本篇长文中只出现在答案所在的
文件里（其他文件的同名注释不会造成歧义）；目标所在的段落确实在本篇长文里。
set.json 只存片段清单、题目、答案和 sha256；run.py 从同一 rev 的 git blob 重建长文。
"""
import argparse
import bisect
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

ENV_READ = re.compile(r'\b(?:getenv|env_zero|env_f|env_value|env_int|env_on)\s*\(\s*"(QWENOX_[A-Z0-9_]+)"')
ENV_ANY = re.compile(r'\bQWENOX_[A-Z0-9_]+\b')
# e = getenv("QWENOX_X"); 下一行 return e ... ? ... : <字面量>;
ENV_DEF = re.compile(
    r'(?:const\s+char\s*\*\s*|auto\s*\*?\s*)(\w+)\s*=\s*(?:std::)?getenv\(\s*"(QWENOX_[A-Z0-9_]+)"\s*\)\s*;'
    r'[ \t]*\n[ \t]*return\s+\1\b[^;\n]*\?[^;\n]*:[ \t]*(-?\d+(?:\.\d+)?)[fFuUlL]*[ \t]*;')
ENV_F = re.compile(r'\benv_f\(\s*"(QWENOX_[A-Z0-9_]+)"\s*,\s*(-?\d+(?:\.\d+)?)[fF]?\s*\)')
LIT = r'(-?(?:0[xX][0-9a-fA-F]+|\d+(?:\.\d+)?))[uUlLfF]*'
CONST_DEFS = [
    re.compile(r'^[ \t]*#define[ \t]+([A-Za-z_]\w*)[ \t]+\(?' + LIT + r'\)?[ \t]*(?://.*)?$', re.M),
    re.compile(r'^[ \t]*(?:static[ \t]+)?(?:inline[ \t]+)?(?:constexpr|const)[ \t]+(?:static[ \t]+)?'
               r'(?:unsigned[ \t]+)?(?:int|unsigned|size_t|uint32_t|uint64_t|int64_t|int32_t|uint16_t|'
               r'uint8_t|float|double|long)[ \t]+([A-Za-z_]\w*)[ \t]*=[ \t]*' + LIT + r'[ \t]*;', re.M),
]
FUNC_DEF = re.compile(
    r'^(?:template[ \t]*<[^>\n]*>[ \t]*)?(?:(?:static|inline|__global__|__device__|__host__|'
    r'__launch_bounds__\([^)\n]*\)|extern|constexpr|\[\[\w+\]\])[ \t]+)*'
    r'[A-Za-z_][\w:<>,]*(?:[ \t]*[*&]+)?[ \t]+[*&]?([A-Za-z_]\w{5,})[ \t]*\([^;{}\n]*\)[ \t]*(?:const[ \t]*)?\{',
    re.M)
SKIP_FUNC = {'main', 'operator', 'return', 'switch', 'sizeof', 'defined'}
NOT_TYPE = {'return', 'else', 'case', 'co_return', 'throw', 'new', 'delete', 'goto', 'do', 'and', 'or', 'not'}
STRICT = {'env_def'}   # 这些题要求名字在本篇只出现在答案文件里（别处注释可能写着过时的默认值）


def decl_lines(text, name):
    """类型开头、紧跟 name( 的行（声明、定义、跨行定义的首行）。"""
    rx = re.compile(r'^[ \t]*((?:[A-Za-z_][\w:<>,]*[ \t*&]+)+)' + re.escape(name) + r'[ \t]*\(', re.M)
    out = []
    for m in rx.finditer(text):
        words = re.findall(r'[A-Za-z_]\w*', m.group(1))
        if words and words[-1] not in NOT_TYPE and words[0] not in NOT_TYPE:
            out.append(text.count('\n', 0, m.start()) + 1)
    return out


def brace_defs(text, name):
    """同一行带 { 的 name(...) —— 含 Cls::name 这种带类名的定义（保守：if (name(x)) { 也算）。"""
    return len(re.findall(r'(?:^|[^\w.>])' + re.escape(name) + r'[ \t]*\([^;\n]*\)[ \t]*(?:const[ \t]*)?\{',
                          text, re.M))


def any_const_defs(text, name):
    """任何写法的定义：#define name ...（不论值）以及 const/constexpr ... name = ...。"""
    n = re.escape(name)
    return (len(re.findall(r'^[ \t]*#[ \t]*define[ \t]+' + n + r'\b', text, re.M)) +
            len(re.findall(r'\b(?:const|constexpr)\b[^;=\n(]*\b' + n + r'[ \t]*(?:=|\{)', text)))


def guarded(text, line, name):
    """#ifndef name / #if !defined(name) 守卫下的定义（系统宏的垫片，模型先验会打架）。"""
    prev = [ln.strip() for ln in text.split('\n')[max(0, line - 4):line - 1] if ln.strip()]
    return any(re.match(r'#[ \t]*if(?:ndef[ \t]+' + re.escape(name) + r'\b|[ \t]+!?[ \t]*defined)', p)
               for p in prev[-1:])


def line_of(starts, pos):
    return bisect.bisect_right(starts, pos)


def index_pool(blobs):
    """全库索引：每个名字的读取/定义位置（文件、行）。"""
    env_reads = defaultdict(list)       # E -> [(path, line)]
    env_defs = defaultdict(list)        # E -> [(path, line_get, line_ret, value)]
    consts = defaultdict(list)          # name -> [(path, line, value)]
    funcs = defaultdict(list)           # name -> [(path, line)]
    for path, text in blobs.items():
        starts = [0] + [m.end() for m in re.finditer('\n', text)]
        for m in ENV_READ.finditer(text):
            env_reads[m.group(1)].append((path, line_of(starts, m.start())))
        for m in ENV_DEF.finditer(text):
            a = line_of(starts, m.start(2))
            env_defs[m.group(2)].append((path, a, line_of(starts, m.end() - 1), m.group(3)))
        for m in ENV_F.finditer(text):
            a = line_of(starts, m.start())
            env_defs[m.group(1)].append((path, a, a, m.group(2)))
        for rx in CONST_DEFS:
            for m in rx.finditer(text):
                consts[m.group(1)].append((path, line_of(starts, m.start(1)), m.group(2)))
        for m in FUNC_DEF.finditer(text):
            name = m.group(1)
            if name not in SKIP_FUNC:
                funcs[name].append((path, line_of(starts, m.start(1))))
    # 同名的定义/声明在全库只能有一处（包括其他形式的定义：常量名不能也是函数名等）
    all_def_names = defaultdict(int)
    for d in (consts, funcs):
        for k, v in d.items():
            all_def_names[k] += len(v)
    return env_reads, env_defs, consts, funcs, all_def_names


def frag_of(frags_by_path, path, a, b):
    for f in frags_by_path.get(path, []):
        if f[1] <= a and b <= f[2]:
            return f
    return None


def mentions(blobs, frag, rx):
    p, a, b = frag
    return bool(rx.search('\n'.join(blobs[p].split('\n')[a - 1:b])))


QTEXT = {
    'env_file': ('上面的源码里，环境变量 `{}` 是在哪个文件里读取的？只回答文件路径。', 'path'),
    'func_file': ('上面的源码里，函数 `{}` 定义在哪个文件？只回答文件路径。', 'path'),
    'const_val': ('上面的源码里，常量 `{}` 的值是多少？只回答数值。', 'num'),
    'env_def': ('上面的源码里，环境变量 `{}` 没有设置时，程序取的默认值是多少？只回答数值。', 'num'),
}


def make_hay(rng, hid, L, cand, frags, frags_by_path, ftok, blobs, per_type):
    """先选目标并把目标段落放到指定深度，再用不含目标名字的段落填满。"""
    targets = []                                   # (type, name, path, a, b, answer, frag)
    used_names, used_frags = set(), set()
    for t, items in cand.items():
        order = items[:]
        rng.shuffle(order)
        k = 0
        for name, path, a, b, ans in order:
            if k == per_type:
                break
            f = frag_of(frags_by_path, path, a, b)
            if f is None or name in used_names:
                continue
            targets.append((t, name, path, a, b, ans, f))
            used_names.add(name)
            used_frags.add(f)
            k += 1
    rx = {name: re.compile(r'(?<![\w])' + re.escape(name) + r'(?![\w])') for t, name, *_ in targets
          if t in STRICT}
    tpath = {name: path for _, name, path, *_ in targets}
    # 目标段落里若出现别的目标名字（且不是那个目标的答案文件），整题作废
    targets = [x for x in targets
               if not any(n != x[1] and tpath[n] != x[6][0] and mentions(blobs, x[6], rx[n]) for n in rx)]
    allrx = {x[1]: re.compile(r'(?<![\w])' + re.escape(x[1]) + r'(?![\w])') for x in targets}
    used_frags = {x[6] for x in targets}
    budget = L - 300 - sum(ftok[f] for f in used_frags)
    filler, n = [], 0
    pool = [f for f in frags if f not in used_frags]
    rng.shuffle(pool)
    for f in pool:
        if n + ftok[f] > budget:
            continue
        # 其他文件的段落里不能出现任何目标名字
        if any(tpath[nm] != f[0] and mentions(blobs, f, r) for nm, r in rx.items()):
            continue
        filler.append(f)
        n += ftok[f]
        if n >= budget - 400:
            break
    # 目标段落按深度均匀插入
    tf = list(used_frags)
    rng.shuffle(tf)
    order = filler[:]
    for j, f in sorted(enumerate(tf), key=lambda x: -x[0]):
        pos = round((j + 0.5) / len(tf) * len(filler))
        order.insert(pos, f)
    text = C.haystack_text(blobs, order)
    offs, o = {}, 0
    for f in order:
        offs[f] = o
        o += len(C.frag_text(blobs, f))
    qs = []
    for t, name, path, a, b, ans, f in targets:
        # 复核：本篇里名字只出现在答案文件
        files = {g[0] for g in order if mentions(blobs, g, allrx[name])}
        if t in STRICT and files != {path}:
            continue
        depth = (offs[f] + len(C.frag_text(blobs, f)) * (a - f[1]) / max(1, f[2] - f[1] + 1)) / o
        qt, kind = QTEXT[t]
        qs.append(dict(id=f'{hid}-{t}-{name}', hay=hid, type=t, kind=kind, q=qt.format(name),
                       answer=ans, name=name, src=[path, a], depth=round(depth, 3),
                       other_files=len(files - {path})))
    qs.sort(key=lambda q: q['depth'])
    return order, text, qs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rev', default='HEAD')
    ap.add_argument('--lengths', default='16384,32768,65536')
    ap.add_argument('--per-len', type=int, default=48, help='每档长度的题数')
    ap.add_argument('--seed', type=int, default=20261015)
    ap.add_argument('--out', default=str(C.HERE / 'set.json'))
    args = ap.parse_args()

    rev = C.git('rev-parse', args.rev).decode().strip()
    paths = C.list_sources(rev)
    blobs = C.read_blobs(rev, paths)
    tk = C.tokenizer()
    env_reads, env_defs, consts, funcs, def_names = index_pool(blobs)
    frags = [f for p in paths for f in C.split_fragments(p, blobs[p])]
    frags_by_path = defaultdict(list)
    for f in frags:
        frags_by_path[f[0]].append(f)
    ftok = {f: len(tk.encode(C.frag_text(blobs, f))) for f in frags}
    print(f'rev {rev[:10]}  {len(paths)} files, {len(frags)} fragments, '
          f'{sum(ftok.values())} tokens in pool')
    cand = {
        'env_file': [(E, v[0][0], v[0][1], v[0][1], v[0][0]) for E, v in env_reads.items()
                     if len({p for p, _ in v}) == 1],
        'func_file': [(k, v[0][0], v[0][1], v[0][1], v[0][0]) for k, v in funcs.items()
                      if len(v) == 1 and def_names[k] == 1 and
                      sum(len(decl_lines(t, k)) for t in blobs.values() if k in t) == 1 and
                      sum(brace_defs(t, k) for t in blobs.values() if k in t) == 1],
        'const_val': [(k, v[0][0], v[0][1], v[0][1], v[0][2]) for k, v in consts.items()
                      if len(v) == 1 and def_names[k] == 1 and len(k) >= 4 and
                      sum(any_const_defs(t, k) for t in blobs.values() if k in t) == 1 and
                      not guarded(blobs[v[0][0]], v[0][1], k)],
        'env_def': [(E, v[0][0], v[0][1], v[0][2], v[0][3]) for E, v in env_defs.items()
                    if len(v) == 1 and sum(f'"{E}"' in t for t in blobs.values()) == 1],
    }
    print('global candidates: ' + ', '.join(f'{t} {len(v)}' for t, v in cand.items()))
    rng = random.Random(args.seed)
    hays, questions = [], []
    for L in map(int, args.lengths.split(',')):
        per_type = max(1, L // 16384)          # 目标段落约占长文 1/4
        for h in range(max(1, args.per_len // (4 * per_type))):
            hid = f'L{L // 1024}K-{h}'
            order, text, qs = make_hay(rng, hid, L, cand, frags, frags_by_path, ftok, blobs, per_type)
            ntok = len(tk.encode(C.prefix_text(text)))
            hays.append(dict(id=hid, L=L, n_tokens=ntok, frags=[list(f) for f in order],
                             sha256=C.sha(text), files=len({f[0] for f in order})))
            questions += qs
            cnt = defaultdict(int)
            for q in qs:
                cnt[q['type']] += 1
            print(f'  {hid}: {ntok} tokens, {len(order)} fragments from {hays[-1]["files"]} files, '
                  f'{len(qs)} questions {dict(cnt)}')
    json.dump(dict(rev=rev, lengths=args.lengths, seed=args.seed, haystacks=hays, questions=questions),
              open(args.out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'wrote {args.out}: {len(hays)} haystacks, {len(questions)} questions')


if __name__ == '__main__':
    main()
