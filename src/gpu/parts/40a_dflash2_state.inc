// ---- DFlash2 draft head state ---------------------------------------------
// Weights come from a separate GGUF sidecar (QWENOX_DFLASH2=<file>). df2_init
// builds device copies straight from its mmap: the projections are fused
// (q|k|v, gate|up, and all five layers' context K|V) and stored as q8g32 by
// default (QWENOX_DF2_WQ=bf16: BF16). Everything here is plain data + device
// pointers; the methods and HIP kernels live in 41_dflash2.inc (included
// after 40_model.inc).
//
// Geometry (DFlash2 / Qwen3.8-Flash-Next, see DFLASH2-HANDOFF.md):
//   5 blocks, hidden 2560, 32 q heads / 8 kv heads, head_dim 128, FFN 7680,
//   conv K=2 groups=16, selector rank 256 top_k 16, block size 8, window 2048.
//   Target features: 5 completed trunk layers [5,15,25,35,45], each averaged
//   over the 4 hyper-connection streams (2560 wide), concatenated to 12800.
static constexpr int kDf2Layers = 5;
static constexpr int kDf2H = 2560;
static constexpr int kDf2NH = 32;
static constexpr int kDf2NKV = 8;
static constexpr int kDf2HD = 128;
static constexpr int kDf2FF = 7680;
static constexpr int kDf2Rank = 256;
static constexpr int kDf2TopK = 16;
static constexpr int kDf2Block = 8;      // block_size: anchor + gamma <= 8
static constexpr int kDf2Window = 2048;  // attention sliding window
static constexpr int kDf2ConvK = 2;
static constexpr int kDf2ConvG = 16;
static constexpr int kDf2Groups = kDf2H / kDf2ConvG;  // 160
static constexpr int kDf2Feat = kDf2Layers * kDf2H;   // 12800
static constexpr int kDf2TapRows = 2048;  // tap ring == window
static constexpr int kDf2VerifyRows = 9;  // gamma + 1 <= 9
static constexpr int kDf2QKV = (kDf2NH + 2 * kDf2NKV) * kDf2HD;  // 6144 fused
static constexpr int kDf2InjRows = 48;   // injection rows per pass (q8 WMMA P<=48)
static constexpr int kDf2TopkRow = 2 * kDf2TopK + 1;  // [16 v][16 id][lse]

// Separate RoPE table: the trunk's global table is initialized for rotary=64,
// DFlash2 rotates the full head_dim (rotary_dim=128, sections [64,0,0,0]).
__device__ double g_df2_rope_tab[64];
__device__ float g_df2_rope_mscale;

static gguf::File* g_gguf_df2 = nullptr;  // QWENOX_DFLASH2 sidecar (mmap alive)

struct Df2LayerW {
  WView qkv;   // [6144][2560]: q (4096) | k (1024) | v (1024)
  WView o;     // [2560][4096]
  WView gu;    // [15360][2560]: gate (7680) | up (7680)
  WView down;  // [2560][7680]
  WView attn_conv_proj, ffn_conv_proj;  // [640][2560]
  // BF16 norm/base tensors converted to f32 once at init (small).
  float* d_attn_norm = nullptr;  // [H]
  float* d_ffn_norm = nullptr;   // [H]
  float* d_q_norm = nullptr;     // [HD]
  float* d_k_norm = nullptr;     // [HD]
  float* d_attn_base = nullptr;  // [2][K][H], side-major
  float* d_ffn_base = nullptr;   // [2][K][H]
};

struct Df2State {
  bool on = false;
  // 0 = off, 1 = prompt prefill (tap ring), 2 = verify (vtap scratch).
  int cap_mode = 0;
  int mask_id = 248077;
  int block_size = kDf2Block;
  int window = kDf2Window;
  float norm_eps = 1e-6f;
  // Verify only the drafted prefix whose running selector probability stays
  // >= conf_th (QWENOX_DF2_CONF; 0 = verify the full gamma).
  float conf_th = 0.5f;
  int trunk_layer[kDf2Layers] = {5, 15, 25, 35, 45};  // completed, zero-based
  // Valid tap/context positions are [max(0, tap_hi - window), tap_hi).
  int tap_hi = 0;

  WView fc;          // [2560][12800]
  WView kv_all;      // [5*2*1024][2560]: per layer K (1024) | V (1024)
  WView sel_hidden;  // [256][2560] (BF16)
  // Raw host BF16 pointers for the selector codebooks (CPU chain walk).
  const uint8_t* sel_pred_host = nullptr;
  const uint8_t* sel_succ_host = nullptr;
  float* d_enc_norm = nullptr;  // [H] HF hidden_norm
  float* d_out_norm = nullptr;  // [H] HF norm

  Df2LayerW lw[kDf2Layers];

  // Target-conditioning ring: tap[pos % 2048][layer][c] (f32, HC mean).
  float* d_tap = nullptr;
  // Verify scratch taps: [row][layer][c] (row < gamma+1).
  float* d_vtap = nullptr;
  // Draft context K/V rings, per block: [layer][pos % 2048][kv head][HD].
  float* d_kctx = nullptr;
  float* d_vctx = nullptr;

  // Injection scratch (<= kDf2InjRows rows).
  float* d_feat = nullptr;  // [rows][12800] gathered tap rows
  float* d_th = nullptr;    // [rows][H] fc output
  float* d_thn = nullptr;   // [rows][H] enc.output_norm(fc output)
  float* d_kvn = nullptr;   // [rows][10240] all-layer context K|V
  // Block-forward scratch (8 rows).
  float* d_x = nullptr;     // [8][H] block hidden
  float* d_norm = nullptr;  // [8][H]
  float* d_conv = nullptr;  // [8][H]
  float* d_qkv = nullptr;   // [8][6144]
  float* d_attn = nullptr;  // [8][NH*HD]
  float* d_o = nullptr;     // [8][H]
  float* d_dyn = nullptr;   // [8][2*K*groups] conv dynamic coefficients
  float* d_gu = nullptr;    // [8][2*FF]
  float* d_ff = nullptr;    // [8][FF]
  float* d_out = nullptr;   // [8][H] post output_norm
  float* d_selh = nullptr;  // [8][rank] selector hidden projection
  float* d_topk = nullptr;  // [8][kDf2TopkRow]
  float* d_topk_part = nullptr;  // [8][chunks][2*16+2] stage-1 partials
  int* d_tok = nullptr;     // [8] block token ids
  float* d_apart = nullptr; // split-key attention partials
  // Keys above this use the split-key attention (QWENOX_DF2_ATTN_SPLIT_MIN).
  int attn_split_min = 256;

  // Host staging reused across rounds.
  float h_topk[kDf2Block * kDf2TopkRow] = {};
  float h_selh[kDf2Block * kDf2Rank] = {};
  float conf[kDf2Block * 3] = {};  // per draft: p_full, q_sel, p_top1
  // QWENOX_DF2_PROF: per-proposal ms [inject wait, 5 blocks, lm_head+topk,
  // host select], reported and cleared per spec loop.
  double prof_ms[4] = {0, 0, 0, 0};
  int prof_n = 0;
  int last_B = 0;  // rows of the last proposal block
  // Experiments (QWENOX_DF2_EMB_BF16 / QWENOX_DF2_LM_BF16): exact HF tables.
  const uint16_t* d_emb_bf16 = nullptr;
  WView lm_bf16{};
  bool lm_bf16_on = false;
};
