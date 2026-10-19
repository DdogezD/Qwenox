# GGUF → i4r 专家补丁

保持 GGUF 文本本体不变，只把路由专家转换成 i4r 补丁，让 prefill 走
`v_wmma_i32_16x16x16_iu4`（`QWENOX_MOE_A4`）。

## 工具

- `tools/gguf_expert_rows.cpp` — 复用 `src/gguf.h:dequant_row`，按
  专家/行 dump `blk.N.ffn_{gate,up,down}_exps` 的 f32。
- `tools/gguf_i4r_patch.py` — 64 专家/块，gate+up 融合成
  `[n,1280,2560]`，H128 旋转 + per-128 scale 的 i4r 编码，按
  `[全部 codes][全部 scales]` 写 hgn payload。
- `tools/bench_i4r_patch.py` — 纯 GGUF / A4=0/2/3 的 prefill、decode
  对比 harness。

helper 不存在时 Python 脚本会自动编译：

```bash
g++ -O2 -std=c++17 -Isrc tools/gguf_expert_rows.cpp -o tools/gguf_expert_rows
```

## 转换

```bash
OMP_NUM_THREADS=1 PYTHONPATH=/path/to/pylib \
python3 tools/gguf_i4r_patch.py \
  --gguf models/<shard1>.gguf \
  --sidecar-src /path/to/reference-v2.hgn \
  --out /path/to/i4r-expert-patch.hgn \
  --workers 12
```

- `--sidecar-src` 提供 192 个 `su/svh` sidecar（专家上基本是 ±1）；
  i4r 权重全部来自 `--gguf`。
- **必须加 `OMP_NUM_THREADS=1`**：否则 BLAS 线程互相抢，明显变慢。
- 数据量：读 46.84 GiB（本模型 48 层 × 512 专家），写 58.01 GiB。
- 实测：12 workers / 679 s，rel-L2 mean 0.09690，max 0.09718。

## 运行

```bash
GGUF=models/<shard1>.gguf
PATCH=/path/to/i4r-expert-patch.hgn
QWENOX_GGUF="$GGUF" \
QWENOX_GGUF_DENSE=1 \
QWENOX_GGUF_EXPERTS=0 \
QWENOX_MOE_A4=3 \
qwenox "$GGUF" "$PATCH" --tokens ... --gen 1
```

- `QWENOX_GGUF_EXPERTS=0` 是关键：专家不再从 GGUF 读，改从 hgn 补丁读；
  dense/router/shared expert 仍来自 GGUF。
- `QWENOX_MOE_A4`：0 = 只换 i4r，不量化激活；1 = gate/up；
  2 = down；3 = 两者。A4 只作用于 prefill。
- 补丁必须覆盖全部 48 层且 gate/up + down 都提供；当前不支持按层混合。

## 启动器配置（start_gguf.sh）

在 `service.conf` 的 GGUF 段加：

```bash
GGUF_I4R_PATCH="$MODEL_DIR/qwen38-swift-i4r-expert-patch.hgn"
MOE_A4=3                 # 2 = 质量/速度折中，3 = 最大 prefill
MEMORY_CAP_GB=100        # 补丁专家 58 GiB pinned；默认 86 可能不够
MIN_AVAILABLE_GB=110     # 按机器实际内存调整
```

启用补丁时建议同时设 `GGUF_MTP_FILE=""`：补丁不含 `mtp.*`；要用 MTP
投机，需要把 v2 hgn 的 31 个 `mtp.*` 张量一起加进补丁。

`start_gguf.sh` 会自动：

- 校验补丁是 `.hgn` 且文件存在（空 = 不用补丁）；
- 把补丁作为第二个位置参数传给引擎：
  `MODEL_ARGS=("$GGUF_FILE" "$GGUF_I4R_PATCH")`；
- `export QWENOX_GGUF_EXPERTS=0`；
- `MOE_A4` 非 0 时由 `serve_common.sh` 导出 `QWENOX_MOE_A4`；
- 补丁已启用但 `MOE_A4=0` 或 `MEMORY_CAP_GB<100` 时打印警告。

`bash start_gguf.sh --check` 应看到：

```text
i4r 专家补丁：.../qwen38-swift-i4r-expert-patch.hgn（QWENOX_GGUF_EXPERTS=0，MOE_A4=3）
ENV QWENOX_GGUF_EXPERTS=0
ENV QWENOX_MOE_A4=3
CMD .../qwenox-engine ...-00001-of-00002.gguf ...-i4r-expert-patch.hgn --serve ...
```

实测启动日志（`MAX_CONTEXT=262144`、4 路、补丁）：`registered mmap 58.01`,
`process RSS 63.98 GiB`, `gpu-accessible committed 82.10 GiB`；`MEMORY_CAP_GB=110`
可以正常启动，默认 86 不建议。

## 实测（gfx1151，单流，真实 qsa-oracle token）

| prompt | 纯 GGUF | A4=2 | A4=3 | A4=3 提升 |
|---:|---:|---:|---:|---:|
| 1K | 1101.1 | 1280.0 | 1383.8 | +25.7% |
| 4K | 1489.5 | 1678.7 | 1845.0 | +23.9% |
| 8K | 1609.4 | 1777.0 | 1969.2 | +22.4% |
| 16K | 1651.6 | 1810.4 | 1985.9 | +20.2% |
| 32K | 1649.1 | 1778.9 | 1948.2 | +18.1% |
| 64K | 1607.1 | 1725.5 | 1893.0 | +17.8% |
| 128K | 1550.8 | 1665.9 | 1822.0 | +17.5% |

单位 tok/s。decode（8K + 64 token）：纯 GGUF 27.7 tok/s，补丁 27.1 tok/s。

质量代价（8×512 KLD，bf16 基准）：

| 配置 | Mean KLD | PPL (qsa 2048) |
|---|---:|---:|
| 纯 GGUF | 0.221430 | 15.1930 |
| A4=2 | 0.281927 | 15.2020 (+0.06%) |
| A4=3 | 0.290159 | 15.9580 (+5.03%) |

转换本身贡献大部分 KLD 增量（A4=0 已 +0.048）；A4=3 在速度和质量之间
取舍最大。上线前建议跑完整 64-chunk KLD 和目标任务评测。

## 限制

- 补丁不含 `mtp.*`；需要 MTP 时从 v2 hgn 补 31 个 `mtp.*` 张量。
- `gg_on()` 是全局开关，不能只补一部分层；per-layer 混合需要改引擎。
- Linux 下补丁专家 `hipHostRegister` 直读；Windows 会拷进 device arena。
- 磁盘：GGUF 77.98 GiB + 补丁 58.01 GiB。
