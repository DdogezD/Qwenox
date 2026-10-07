# gfx1151-engine vs gufo：prefill 差距分析（2026-09-25）

gufo：`gufo-main/src/models/qwen38_flash_next`，同一模型，权重为 Unsloth UD-Q4_K_XL GGUF，
prefill chunk 固定 2048，单 stream，prefill 不用 hipGraph。
它公布的 pp2048@depth（预热，HTTP）：d0 1628 / d4K 1523 / d8K 1499 / d16K 1457 / d32K 1422 / d64K 1304 tok/s。

## 1. 同口径实测（我们，生产 env，32K prompt，远程 logs/cmp_c*.log）

| chunk | 每 chunk tok/s | 说明 |
|---|---|---|
| 2048 | 745(冷) 830 803 802 793 787 800 776 783 762 779 777 771 790 785 762 | 约 800，与 gufo 同口径 → **gufo ≈ 2.0×** |
| 4096 | 852(冷) 920 913 919 912 894 896 903 | |
| 16384（生产） | 1194(冷) 1231 | 即使用我们最优配置，gufo@2048 仍快 1.15–1.3× |

我们的性能严重依赖大 chunk；gufo 在 2048 就到了 1400–1600。

## 2. 逐组件对照（ms/token，kernel 时间）

chunk=2048，8K prompt（logs/kt_c2048.ktrace，合计 10.29 s / 8192 tok）对比 gufo d0 profile：

| 组件 | 我们 | gufo | 差 |
|---|---|---|---|
| 路由 MoE：k_moe_w4_up 3476 ms + k_moe_w4_down 1963 + reduce 209 + router 52 | **0.696** | 0.233 | **+0.463（约 78%）** |
| Dense 投影（k_gemm_wmma + router/iproj Lt + q4cp 反量化；含 HC up/down、shared） | 0.301 | 0.240（dense 0.193 + HC down/shared 0.047） | +0.061 |
| 注意力 + indexer（我们是 0–8K 深度平均） | 0.096 | 0.037 | +0.059 |
| HC combine + scatter_norm | 0.075 | 0.070 | ≈ |
| GDN（fused + conv + gatednorm + l2norm） | 0.070 | 0.063 | ≈ |
| 合计 | 1.256 | 0.665 | 0.591 |

chunk=16384，32K（logs/dab_kt_sp_32k.ktrace，25.72 s / 32768 tok）：MoE 0.291（Lt GEMM 20 种 tile 共 6.52 s，
约 24 TFLOPS；外围 deq_bf16 1.35 s + gather 0.73 + reduce 0.46 + silu 0.25 + router 0.21 ≈ 3.0 s，占 31%），
dense 0.246，注意力 0.092，HC 0.069，GDN 0.068。
> 更正：之前说“我们 MoE 更便宜”是错的，那次只算了 2 种 Lt tile。

## 3. 根因：小 chunk 的 MoE kernel 没有用矩阵核

- chunk < `GDEC_MOE_LT_MIN`(4096) 时走 `k_moe_w4_up/down`（23_kernels_moe_w4.inc）。
- 它在 LDS 里存反量化后的 bf16，然后做**标量 `v_dot2_f32_bf16`**。注释里自己也写着受 LDS 带宽限制。
- 实测：约 7 TFLOPS；每层 1.26 GB 权重读了 28 ms，折合 45 GB/s。**既不在算力上限，也不在带宽上限。**
- gufo 的 `RoutedF16GEMMKernel`（kernels.hip.cpp:3637）做法：
  - 原始 Q4_K/Q5_1 码字以打包形式放进 LDS，两个 K 块只占 11–12 KB，每个 WGP 驻留 5 个 block。
  - 每个 wave 读出自己那 16 行后解码：`perm` 加 0x6400 魔数得到 F16，再一次 `hfma2` 套 scale/bias。
  - F16 WMMA，F32 在整个 K 上累加，不做逐块的 epilogue。
  - gate/up 成对（wave 0-3 算 gate、4-7 算 up），epilogue 里做 SwiGLU，直接写 down 的 F16 输入。
  - down 写 F16 `[token][slot][hidden]`；带权求和放进下一步的 HC combine。
  - 由 host 根据 D2H 回传的专家计数建 tile 表（每层同步一次），tile 按专家分 48/64/128 行三档。
  - 结果：整个 MoE 约 20 TFLOPS。

## 4. 可借鉴点与预估收益（按优先级）

1. **把融合 MoE kernel 改成 WMMA 版（最大杠杆）**
   - 做法：Q4C-P 码字留在 LDS，读出后查 16 项码本转 bf16/f16，再做 WMMA 16x16x16。
   - gate/up 成对并在 epilogue 做 SwiGLU；down 带 router 权重输出。
   - 小 chunk 和大 chunk 用同一个 kernel，这样就不再需要 deq_bf16 + Lt + gather/silu 那条路径。
   - 设计点：码本不是均匀量化，不能直接照搬 gufo 的魔数技巧。可以沿用 cb_pairs（LDS 表），或者把 16 项码本放在寄存器里用 perm 查表。
   - 预估：
     - chunk=2048：MoE 0.70 → 约 0.25，整体约 800 → 约 1200 tok/s（+50%）。
     - 生产 16K chunk：MoE 0.29 → 约 0.20，1231 → 约 1370（+11%）。
     - 之后可以重新扫一遍 chunk 大小：小 chunk 能省约 11 GB workspace，短 prompt 的首 token 延迟也更好。
2. **HC gate 融合（#51，原型已 bit-exact，up GEMM 加 combine 快 2.1×）**：每 32K 省约 1 s，+3.5–4%。gufo 的 HcMix 就是这么做的，gate logits 不落显存。
3. **在生产者 kernel 的 epilogue 里直接写出下一个 GEMM 的输入格式**
   - gufo 的 HcCombine 同时写 F32 残差、F16 xn 和 tiled Q8 xn。
   - QKV GEMM 的 epilogue 里做 q/k RMSNorm、RoPE、KV 写入；SSM GEMM 的 epilogue 里做 conv4 + SiLU。
   - 我们对应的独立 kernel：k_l2norm_qk_b、k_gdn_conv_b、k_qsa_qsplit、k_sigmoid_gate、k_f32_to_bf16_v4、k_dequant_q4cp_bf16。它们在 16K chunk 合计约 1.4 s/32K，预估 +3–5%。
   - **2026-09-28 已实测否决（TUNING-WINDOWS §3.10）**：两个 bit-exact 融合
     组件级 -262 ms，端到端 6 对 A/B 中位数 ~0（≤0.3%，不超噪声）。kernel
     trace/事件对是小 kernel 的孤立时间，原位被相邻大 kernel 的边界重叠
     吸收，+3-5% 的估计不可恢复。同类 GEMM-epilogue 候选一并否决。
4. **Dense**：chunk=2048 时我们 0.301 对 gufo 0.240。
   - k_dequant_q4cp_bf16 每次调用都要反量化一遍（c2048 下 108 ms/8K）。
   - gufo 是在寄存器里把 Q8_0 解码后直接进 LDS。
   - 对 K 大 N 小的形状（2560×6144、320×10240），gufo 用 int8 WMMA 更快。
5. **注意力（深度越深占比越大）**
   - gufo：F16 KV；一个 block 负责 4 query × 12 head，把 mask 的并集压成 LDS 列表，按 16-key 的 tile 跳过。
   - 可以与 k_qsa_wmma 逐项对照（它在 32K 占 6.9%）。
   - **2026-09-28 已证伪**：同结构实现 `GDEC_QSA_UNION` 复测为中性（§6 末段），不再是候选。

## 5. 已完成与否决的 Gufo 借鉴实验

- **HC mix-up/combine Epi fusion：接受**。复用已有 `k_gemm_wmma<...,Epi=1>`，
  原经 `GDEC_GR_WMMA_EPI=1` opt-in，将 HC up 投影与 combine 合并；Windows
  固定 32K CLI A/B 的主 chunk 中位数约 1167.7→1194.6 tok/s（约 +2.2%），
  生成输出逐字节一致，已提交为 `7c64370`。2026-09-28 起代码默认开启
  （`GDEC_GR_WMMA_EPI=0` 关闭）：在含 PLE 预取的二进制上复验 3 对 32K
  A/B 中位数 +2.3%、ids 逐位一致，详见 `TUNING-WINDOWS.md §3.8`。
- **Adaptive MTP effective gamma：否决**。`GDEC_MTP_ADAPTIVE=1` 在固定
  gamma=3 之上没有收益，三次 decode 约 29.6–29.7 vs 29.8–30.0 tok/s；
  输出对拍通过但约慢 0.8–1.3%，代码已撤销。
- **QSA producer epilogue：否决**。临时 `GDEC_QSA_PRODUCER_EPI=1` 将 K
  normalize/RoPE 与 BF16 K/V producer 合并；三次组共 9 个主 chunk 的中位数
  1168.3 vs 1167.7 tok/s，约 +0.05%，属于噪声；输出对拍通过，代码已撤销。

详细环境、原始日志标签和 gate 结论见 `TUNING-WINDOWS.md §3.3`。

## 6. 已验证方向与剩余差距

已完成 standalone production-equivalent F16 prototype gate：
`tools/moe_wmma_proto.cu` 新增独立的 F16 activation/weight LDS staging，使用
生产 LUT-WMMA 的 weight-A、activation-B、`BN+1` activation stride、F16
codebook/scale multiply 和 F32 accumulation；生产文件未修改。Windows
TheRock executable 在 synthetic routed Q4C-P、rows=1280、cols=2560 下通过
uniform `maxrel=9.362e-4`（19.4 TFLOPS）和 skew `maxrel=9.400e-4`
（18.9 TFLOPS），无 non-finite 输出。K=64 合法最小 tile 也通过；K<64 不属于
生产 LUT-WMMA 合同，不能作为 gate。

此前 DM1/DM2 的 `maxrel=0.7275` 仍是 harness reference mismatch：它们复现旧
`k_moe_w4_up` 的 BF16 codebook-pair + BF16 scale multiply，而原 reference 只
复现 FP32 product 后转 BF16。该结果已用独立匹配 reference 修正，但与当前
production F16 path 分开记录。F16 prototype 目前仍只覆盖 synthetic up-GEMM，
尚未在 real HGN Q4CP 上比较 up/down/total，也未接入生产 dispatch；下一步必须
先以 `tools/moe_lut_test.cu` 的 real-HGN LUT-WMMA 为基线再测，不能凭 synthetic
TFLOPS 替换生产路径。

**Real-HGN 对比已完成，routed WMMA prototype 方向否决（2026-09-28）。**
`tools/moe_lut_test.cu --cand up-f16`（k_moe_lut_f16_cand = prototype 的
per-expert block + block 内 slot 循环 + 生产 gate/up 配对与 silu epilogue）
在 heretic.hgn layer 0 上与生产 `moe_lut_up` 交替 A/B：正确性逐位一致
（up rel-L2 max 5.281e-04，双臂 RESULT PASS），但 up 中位数在所有 P 下
更慢——2048：8.960 vs 8.588 ms（+4.3%）；4096：15.476 vs 13.986
（+10.7%）；8192：27.752 vs 25.877（+7.2%）。差距来自调度（MoeTile
逐块分发的负载均衡优于每专家一个 block 的循环），不是数值。结论：现有
生产 LUT-WMMA 已是该设计空间的优选点，Gufo 式 routed WMMA 不再作为
prefill 差距的修复方向；后续候选目标按 `TUNING-WINDOWS.md §3.6` 之后的
阶段另选（producer epilogue、dense 小 chunk、PLE 热行缓存等），各自走
独立 A/B。注意力形状特化已于 2026-09-28 复测否决（见 §4.5 与本节末段）。

**PLE 泵流水化：接受（2026-09-28，§2 表外的组件）。** §2 的组件表没有
PLE 段——它是 47.7 GiB FP8 n-gram 表的随机 160B 行磁盘泵，生产 16K
phase 分解实测占 11.4%（GPU 每块空等 ple_wait 570–990 ms）。行地址
只依赖 token id，下一 chunk 的收集可完全藏进本 chunk 层计算：双
staging + 单 flight 预取后，32K 端到端 28.54→26.24 s（+8.8%），稳态块
1153–1176→1306–1334 tok/s（+12.5–13.4%），生成 ids 与 MTP spec 行为
逐位一致。详见 `TUNING-WINDOWS.md §3.7`。至此 16K 口径我们对 gufo
的差距约为 1300 vs 1457（0.90×）。

**注意力结构方向：否决（2026-09-28，复测确认）。** §4.5 推测 gufo 的
注意力优势来自"一个 block 负责 4 query × 12 head + mask 并集"。该结构
我们已有实现（`GDEC_QSA_UNION=1`，k_qsa_union_merge + k_qsa_wmma_u），
2026-09-22 测 -3.6% 否决；在当前栈（LUT-WMMA + PLE 预取 + EPI）上
去混淆复测为**中性**（32K 端到端 25.80 vs 25.89 s，噪声内，
`TUNING-WINDOWS.md §3.9`）。K 读分摊不兑现：相邻 token 的逐 token
重读大多命中 L2，merge kernel 与更大 block 占用吃掉差额。gufo 形状的
多 query 块结构在本硬件上无利可图；§2 表里的注意力差（0.096 vs
0.037）还含口径问题（我们是 0–8K 深度平均、gufo 是 d0），真实差距
远小于表观值。注意力不再作为 prefill 差距的修复方向。

剩余差距主要改述为：kernel 数量/口径、测量口径（gufo 表为预热后、
我们 pp_prod 为冷启动），以及 dense 小 chunk 反量化（仅 chunk=2048
口径）。

## 7. 其他说明

- **质量**：gufo 用的 Q4_K_XL，KLD 约 0.049；我们引擎是 0.163（KLD.md）。gufo 在速度和质量上都占优。imatrix 重量化仍然要做。
- **decode**：gufo 文档写的 tg 约 26 tok/s；我们不开投机是 31.9，开 MTP 约 55，decode 不落后。
- **测量口径**：gufo 表里的数都是预热后测的；我们的 pp_prod 是冷启动单次，第一个 chunk 偏低约 5–8%。
