"""codeneedle 公共部分：读源码 blob、切片、重建长文、拼 prompt、判分。

build.py（出题）和 run.py（运行）都只通过这里拼长文，保证两边逐字节一致；
set.json 里记录每篇长文的 sha256，run.py 重建后先核对。
"""
import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / 'tools'))

EXTS = ('.cpp', '.h', '.inc', '.hip', '.cu', '.hpp', '.c')
# 生成的表、图标、第三方代码不当素材
EXCLUDE = re.compile(r'(^src/api/vendor/|static_gen\.inc$|unicode_tables\.inc$|_icon\.inc$|icon\w*\.inc$)')
FRAG_CHARS = 3000          # 每段约 1K token（代码约 2.9 字符/token）
INTRO = ('下面是 qwenox 推理引擎的部分源码。每段以 `// ===== file: 路径 (lines a-b) =====` 开头，'
         '段落顺序已打乱，同一个文件可能出现多段。\n\n')
IM_END = 248046
ENDOFTEXT = 248044


def tokenizer():
    from qwentok import Tokenizer
    for d in (os.environ.get('TOK_DIR'), str(ROOT / 'models/tokenizer'), str(ROOT / 'data/tokenizer')):
        if d and (Path(d) / 'tokenizer.json').is_file():
            return Tokenizer(d)
    raise SystemExit('tokenizer.json not found (TOK_DIR / models/tokenizer / data/tokenizer)')


def git(*args):
    return subprocess.run(['git', '-C', str(ROOT), *args], check=True,
                          capture_output=True).stdout


def list_sources(rev):
    names = git('ls-tree', '-r', '--name-only', rev, '--', 'src').decode().split('\n')
    return sorted(n for n in names if n.endswith(EXTS) and not EXCLUDE.search(n))


def read_blobs(rev, paths):
    """一次 cat-file --batch 读完所有 blob；统一成 LF、UTF-8。"""
    inp = ''.join(f'{rev}:{p}\n' for p in paths).encode()
    out = subprocess.run(['git', '-C', str(ROOT), 'cat-file', '--batch'], input=inp,
                         check=True, capture_output=True).stdout
    res, i = {}, 0
    for p in paths:
        nl = out.index(b'\n', i)
        hdr = out[i:nl].split()
        if len(hdr) < 3 or hdr[1] != b'blob':
            raise SystemExit(f'git cat-file: bad object for {p}: {out[i:nl]!r}')
        n = int(hdr[2])
        body = out[nl + 1:nl + 1 + n]
        i = nl + 1 + n + 1
        res[p] = body.decode('utf-8', errors='replace').replace('\r\n', '\n')
    return res


def split_fragments(path, text):
    """按行切成约 FRAG_CHARS 的段，尽量在空行处断开。返回 [(path, a, b)]，行号从 1 开始、闭区间。"""
    lines = text.split('\n')
    if lines and lines[-1] == '':
        lines.pop()
    frags, a, n = [], 0, len(lines)
    while a < n:
        size, b, last_blank = 0, a, -1
        while b < n and size + len(lines[b]) + 1 <= FRAG_CHARS:
            size += len(lines[b]) + 1
            if not lines[b].strip():
                last_blank = b
            b += 1
        if b == a:          # 单行超长
            b = a + 1
        elif b < n and last_blank > a + (b - a) // 2:
            b = last_blank + 1
        frags.append((path, a + 1, b))
        a = b
    return frags


def frag_text(blobs, frag):
    path, a, b = frag
    lines = blobs[path].split('\n')
    body = '\n'.join(lines[a - 1:b])
    return f'// ===== file: {path} (lines {a}-{b}) =====\n{body}\n\n'


def haystack_text(blobs, frags):
    return ''.join(frag_text(blobs, f) for f in frags)


def sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def prefix_text(hay):
    return '<|im_start|>user\n' + INTRO + hay + '\n'


def suffix_text(question):
    return (question + '\n只输出答案本身，不要解释。<|im_end|>\n'
            '<|im_start|>assistant\n<think>\n\n</think>\n\n')


def encode_plain(tk, text):
    """纯 BPE，不匹配特殊 token：源码里的 `<|vision_start|>`、`<|im_end|>` 等字面量按普通字符编码，
    不会变成控制 token（否则引擎会当成图像占位符拒绝，或者把对话结构截断）。"""
    import unicodedata
    from qwentok import _split, _B2U
    ids = []
    for ptok in _split(unicodedata.normalize('NFC', text)):
        for t in tk._bpe(''.join(_B2U[b] for b in ptok.encode('utf-8'))):
            ids.append(tk.vocab[t])
    return ids


def prefix_ids(tk, hay):
    """<|im_start|>user\\n + 说明 + 长文（长文按普通文本）。"""
    return tk.encode('<|im_start|>user\n') + encode_plain(tk, INTRO + hay + '\n')


def suffix_ids(tk, question):
    return (encode_plain(tk, question + '\n只输出答案本身，不要解释。') +
            tk.encode('<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'))


def special_ids(tk):
    return set(tk.special.values())


# ---------------- 判分 ----------------
# 独立的数字：前后不粘 ASCII 字母/数字/下划线/点（避开 Q6W_BLOCKS 这类名字里的数字），允许 f/u/l 后缀
_NUM = re.compile(r'(?<![A-Za-z0-9_.])(-?(?:0[xX][0-9a-fA-F]+|\d+(?:\.\d+)?(?:[eE][-+]?\d+)?))[fFuUlL]*'
                  r'(?![A-Za-z0-9_])(?!\.\d)')


def parse_num(s):
    s = s.strip().rstrip('fFuUlL')
    try:
        return int(s, 16) if s.lower().lstrip('-').startswith('0x') else float(s)
    except ValueError:
        return None


def first_line(out):
    for ln in out.strip().split('\n'):
        ln = ln.strip()
        if ln:
            return ln
    return ''


def judge(q, out, basenames):
    """返回 (是否正确, 抽出来的答案串)。basenames: 本篇长文里 basename→路径数。"""
    ln = out.strip()
    if q['kind'] == 'num':
        m = _NUM.search(ln)
        if not m:
            return False, first_line(out)[:60]
        v, want = parse_num(m.group(1)), parse_num(q['answer'])
        return (v is not None and want is not None and abs(v - want) <= 1e-9 * max(1, abs(want))), m.group(1)
    m = re.search(r'[A-Za-z0-9_./\\-]+\.(?:cpp|h|inc|hip|cu|hpp|c)\b', ln)
    if not m:
        return False, ln[:60]
    got = m.group(0).replace('\\', '/').lstrip('./')
    want = q['answer']
    if got == want or want.endswith('/' + got) and basenames.get(got.rsplit('/', 1)[-1], 0) == 1:
        return True, got
    return False, got
