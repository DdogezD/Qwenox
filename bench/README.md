# bench — 任务级评测

## codeneedle：长上下文代码捞针

用引擎自己的源码当长文，问答案唯一、能自动核对的问题，比较两种引擎设置下模型读长输入的准确度。主要用来回答"某个只影响 prefill 的改动（比如 `MOE_A4`）在实际任务里能不能被感知"。

### 为什么这样测

- **只考输入。** 答案只有十几个 token，几乎全部计算都在 prefill 上（16K–64K 的长文），改动的误差能覆盖约 99% 的上下文。像"一句话生成一个程序"这类题，prompt 很短、推理全走 decode，测不出 prefill 的改动。
- **题目不靠显眼的针。** 素材是真实源码，周围全是相似的名字（上百个 `QWENOX_*` 环境变量、成组的 `ADL_PMLOG_*` 常量），而且段落顺序打乱、同一文件可能被切成多段，靠字面匹配很难直接找到答案。
- **答案唯一。** 出题时在全库做唯一性校验，并用另一套纯文本搜索独立复核。

### 四类题

| 类型 | 例子（64K） | 答案 |
|---|---|---|
| `env_file` | 环境变量 `QWENOX_GDN_WAVE` 是在哪个文件里读取的？ | `src/gpu/parts/40_model.inc` |
| `func_file` | 函数 `heavy_gate` 定义在哪个文件？ | `src/gpu/parts/51_host_cfg.inc` |
| `const_val` | 常量 `ADL_PMLOG_ASIC_POWER` 的值是多少？ | `23` |
| `env_def` | 环境变量 `QWENOX_FORK_DUMP` 没有设置时，程序取的默认值是多少？ | `0`（要读懂 `getenv` 后面的三元表达式） |

筛题规则：
- 环境变量在全库只在一个文件里被读取。其他文件里的 `setenv` / `set_env_value`（写入）和注释允许出现，模型要分清是读还是写。
- 函数在全库只有一处定义：声明、跨行定义、带类名的同名定义都会被排除；其他文件里的调用允许出现。
- 常量只有一处定义（任何写法，包括条件编译的另一个分支）；排除 `#ifndef` 守卫下的系统宏垫片。
- 默认值题最严格：变量名在全库只出现在一个文件里，避免别处注释写着过时的默认值。

### 题量

`set.json` 由 `build.py --rev b27bb8b` 生成：每档长度 48 题，共 144 题、21 篇长文，合计约 58 万 token。

| 长度 | 篇数 | 每篇题数 |
|---|---:|---:|
| 16K | 12 | 4 |
| 32K | 6 | 8 |
| 64K | 3 | 16 |

目标段落约占长文的 1/4，按深度均匀插入，其余用随机段落填满。`set.json` 只存片段清单、题目、答案和 sha256；运行时从同一 rev 的 git blob 重建长文，并核对 sha256。

### 怎么跑

Windows（Git Bash，先停服务，约 30–40 分钟）：

```sh
bash bench/codeneedle/ab_win.sh            # A：MOE_A4 不设，B：MOE_A4=3
A4=1 bash bench/codeneedle/ab_win.sh       # 只量化 gate/up
ONLY=L16K bash bench/codeneedle/ab_win.sh  # 只跑 16K，几分钟看个大概
```

两轮都直连裸引擎（8732），`QWENOX_KVSNAP=0`，所以两轮之间不会复用对方的 prompt KV。RAM 检查点开着：每篇长文只 prefill 一次，之后每题只 prefill 几十个 token 的问题部分。问题部分超过 16 个 token，同样走 prefill 路径，所以也受 A4 影响，和真实使用一致。最后一行是 `CODENEEDLE PASS|FAIL`。

离线自测（不需要 GPU，十几秒）：

```sh
python bench/codeneedle/selftest.py        # 判分用例 + 假引擎跑通 run/cmp，最后一行 SELFTEST PASS|FAIL
```

### 每题怎么答

- **1 次贪心**：主要看准确率。
- **4 次全分布采样**：`temp 1, top_k 0, top_p 1`，带 `SEEDED` 和 `LOGPROBS`，seed 由题号决定，两轮相同。

两轮用的随机数完全一样，只有改动把概率分布推过了那个随机数，采出来的答案才会不同。所以采样翻转的多少直接反映两个分布差了多少；两轮采出同一个答案时，它的 logprob 差就是模型"把握"的变化。

### 结果怎么读

`cmp` 按 16K / 32K / 64K / 全部 给一行：

| 列 | 含义 |
|---|---|
| greedy A / B | 贪心准确率 |
| A>B / B>A | A 对 B 错、A 错 B 对的题数 |
| p | 精确 McNemar（符号检验）双侧 p 值 |
| B-A 95% CI | 准确率差的置信区间（百分点） |
| same ans | 两轮贪心输出逐字相同的比例 |
| sample A / B 及其翻转 | 同上，按"题 × seed"计 |
| dlogp | 两轮采出同一答案时 logP(B) − logP(A) 的均值，负数表示 B 更没把握 |

- **FAIL**：运行出错，或者 B 比 A 差、且 p < 0.05。
- **PASS**：没测到可感知的退化。这时应该看置信区间来下结论。比如全部题的区间是 [−4, +3]，结论应写成"准确率损失不超过约 4 个百分点"，而不是"没有影响"。
- 如果某档 A 的准确率高于 97% 或低于 20%，表格下面会提示：这一档翻转很少，信息量有限，主要看 dlogp。

### 文件

| 文件 | 作用 |
|---|---|
| `codeneedle/build.py` | 出题：`python bench/codeneedle/build.py [--rev REV] [--lengths 16384,32768,65536] [--per-len 48]` |
| `codeneedle/run.py` | `run --tag T` 跑一轮，结果写 `logs/codeneedle/T.json`；`cmp --a A --b B` 对比 |
| `codeneedle/common.py` | 读 blob、切段、拼 prompt、判分（build/run 共用，保证长文逐字节一致） |
| `codeneedle/ab_win.sh` | 一键 A/B |
| `codeneedle/selftest.py` | 离线自测 |
| `codeneedle/set.json` | 题集（源码 rev b27bb8b） |

分词器按 `TOK_DIR` → `models/tokenizer` → `data/tokenizer` 的顺序查找。
