// gg_lin_decomp.cpp — CPU 对拍：GGUF IQ 权重 per-128 int4 线性化的误差分解。
// 验证 29_kernels_moe_gg_iu4.inc 的 kernel 是否实现了其设计数学，并把 proto
// 测到的 A4-vs-F16 输出误差分解为 权重线性化 / int4 激活 两部分。
//
//   gg_lin_decomp FIRST_SHARD.gguf [layer] [div] [experts] [tokens]
//
// 线性化数学（与 kernel 的 qtab/fetch_w 一致）：
//   128-K 组 = 4 个 32 元素子块；md = max|d_b|；sf = md*kvmax/div
//   l = clamp(rint(|d_b|/md*15),0,15)/16 档；q = clamp(rint(w/|d_b|*l*div/kvmax),-8,7)
//   ŵ = q * sf        （sg*r*v 恒等于 w/|d_b|，无需码表）
// 激活量化与 k_quant_a4 一致（5 候选 MSE-best clip，s = cand*amax/7）。
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include "../src/gguf.h"

static float f16_to_f32(uint16_t h) {
  const int sign = (h >> 15) & 1, exp = (h >> 10) & 31, mant = h & 1023;
  float v;
  if (exp == 0) v = ldexpf((float)mant, -24);
  else if (exp == 31) v = mant ? NAN : INFINITY;
  else v = ldexpf((float)(mant | 1024), exp - 25);
  return sign ? -v : v;
}

// 子块（32 元素）的 GGUF 块内 scale d_b；ib = 行内全局子块索引。
static float sub_scale(uint32_t t, const uint8_t* row, int ib) {
  uint16_t dh;
  if (t == 20) {  // IQ4_NL: 18B/32
    memcpy(&dh, row + (size_t)ib * 18, 2);
    return f16_to_f32(dh);
  }
  if (t == 23) {  // IQ4_XS: 136B/256
    const int sb = ib >> 3, sub = ib & 7;
    const uint8_t* blk = row + (size_t)sb * 136;
    uint16_t sh;
    memcpy(&dh, blk, 2);
    memcpy(&sh, blk + 2, 2);
    const uint8_t* sl = blk + 4;
    const int ls = ((sl[sub >> 1] >> (4 * (sub & 1))) & 0xF) | (((sh >> (2 * sub)) & 3) << 4);
    return f16_to_f32(dh) * (float)(ls - 32);
  }
  if (t == 21) {  // IQ3_S: 110B/256
    const int sb = ib >> 3, sub = ib & 7;
    const uint8_t* blk = row + (size_t)sb * 110;
    memcpy(&dh, blk, 2);
    const int sc4 = (blk[106 + (sub >> 1)] >> (4 * (sub & 1))) & 0xF;
    return f16_to_f32(dh) * (float)(1 + 2 * sc4);
  }
  fprintf(stderr, "unsupported type %s\n", gguf::type_name(t));
  exit(2);
}

// 一行权重按 kernel 数学线性化；同时返回权重 rel-L2 的分子/分母。
static void linearize_row(uint32_t t, const uint8_t* raw, const float* w, double* wl, int K,
                          float div, double& num, double& den) {
  const double kvmax = t == 21 ? 15.0 : 127.0;
  for (int g = 0; g < K / 128; ++g) {
    float db[4];
    double md = 0;
    for (int j = 0; j < 4; ++j) {
      db[j] = sub_scale(t, raw, g * 4 + j);
      md = fmax(md, (double)fabsf(db[j]));
    }
    const double sf = md * kvmax / div;
    for (int j = 0; j < 4; ++j) {
      const double ad = fabs(db[j]);
      int l = md > 0 ? (int)rint(ad / md * 15.0) : 0;
      l = l < 0 ? 0 : l > 15 ? 15 : l;
      const double r = l / 15.0;
      for (int i = 0; i < 32; ++i) {
        const int c = g * 128 + j * 32 + i;
        const double wv = w[c];
        double q = 0;
        if (ad > 0 && sf > 0) q = rint(wv / ad * r * div / kvmax);
        q = q < -8 ? -8 : q > 7 ? 7 : q;
        wl[c] = q * sf;
        num += (wl[c] - wv) * (wl[c] - wv);
        den += wv * wv;
      }
    }
  }
}

// k_quant_a4 复刻：返回重建值 xq（= int4 * scale）。
static void quant_a4(const double* x, double* xq, int K) {
  for (int b = 0; b < K / 128; ++b) {
    double amax = 0;
    for (int i = 0; i < 128; ++i) amax = fmax(amax, fabs(x[b * 128 + i]));
    double best_s = 0, best_e = HUGE_VAL;
    if (amax > 0) {
      const double cand[5] = {1.0, 0.9, 0.8, 0.7, 0.6};
      for (int c = 0; c < 5; ++c) {
        const double sc = cand[c] * amax / 7.0;
        double e = 0;
        for (int i = 0; i < 128; ++i) {
          double q = rint(x[b * 128 + i] / sc);
          q = q < -8 ? -8 : q > 7 ? 7 : q;
          const double d = x[b * 128 + i] - q * sc;
          e += d * d;
        }
        if (e < best_e) {
          best_e = e;
          best_s = sc;
        }
      }
    }
    for (int i = 0; i < 128; ++i) {
      double q = best_s > 0 ? rint(x[b * 128 + i] / best_s) : 0;
      q = q < -8 ? -8 : q > 7 ? 7 : q;
      xq[b * 128 + i] = q * best_s;
    }
  }
}

struct Tensor2 {
  uint32_t type;
  int rows, K;
  const uint8_t* raw;  // 某专家起点
  std::vector<float> w;    // 精确 dequant [rows][K]
  std::vector<double> wl;  // 线性化 [rows][K]
};

static void load_tensor(const gguf::Tensor& t, int e, int rows, int K, float div,
                        Tensor2& out, const char* name) {
  out.type = t.type;
  out.rows = rows;
  out.K = K;
  out.raw = t.data + (size_t)e * rows * t.row_bytes();
  out.w.resize((size_t)rows * K);
  out.wl.resize((size_t)rows * K);
  double num = 0, den = 0;
  for (int r = 0; r < rows; ++r) {
    const uint8_t* rp = out.raw + (size_t)r * t.row_bytes();
    gguf::dequant_row(t.type, rp, &out.w[(size_t)r * K], K);
    double rn = 0, rd = 0;
    linearize_row(t.type, rp, &out.w[(size_t)r * K], &out.wl[(size_t)r * K], K, div, rn, rd);
    num += rn;
    den += rd;
  }
  printf("  %s %-7s [%d x %d] 权重线性化 rel-L2 = %.4f\n", name, gguf::type_name(t.type), rows,
         K, sqrt(num / fmax(den, 1e-30)));
}

static void matvec(const Tensor2& T, const double* x, double* y) {
  for (int r = 0; r < T.rows; ++r) {
    double s = 0;
    for (int c = 0; c < T.K; ++c) s += T.w[(size_t)r * T.K + c] * x[c];
    y[r] = s;
  }
}

static void matvec_lin(const Tensor2& T, const double* x, double* y) {
  for (int r = 0; r < T.rows; ++r) {
    double s = 0;
    for (int c = 0; c < T.K; ++c) s += T.wl[(size_t)r * T.K + c] * x[c];
    y[r] = s;
  }
}

static double rel_l2(const std::vector<double>& a, const std::vector<double>& b) {
  double num = 0, den = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    num += (a[i] - b[i]) * (a[i] - b[i]);
    den += b[i] * b[i];
  }
  return sqrt(num / fmax(den, 1e-30));
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s FIRST_SHARD.gguf [layer] [div] [experts] [tokens]\n", argv[0]);
    return 2;
  }
  const int layer = argc > 2 ? atoi(argv[2]) : 0;
  const float div = argc > 3 ? (float)atof(argv[3]) : 9.0f;
  const int nexp = argc > 4 ? atoi(argv[4]) : 4;
  const int ntok = argc > 5 ? atoi(argv[5]) : 16;
  gguf::File f(argv[1]);
  char nm[128];
  snprintf(nm, sizeof nm, "blk.%d.ffn_gate_exps.weight", layer);
  const gguf::Tensor& tg = f.at(nm);
  snprintf(nm, sizeof nm, "blk.%d.ffn_up_exps.weight", layer);
  const gguf::Tensor& tu = f.at(nm);
  snprintf(nm, sizeof nm, "blk.%d.ffn_down_exps.weight", layer);
  const gguf::Tensor& td = f.at(nm);
  const int D = (int)tg.ne[0], MID = (int)tg.ne[1], E = (int)tg.ne[2];
  printf("layer %d: gate/up %s [%d x %d x %d], down %s, div %.2f, experts %d, tokens %d\n", layer,
         gguf::type_name(tg.type), D, MID, E, gguf::type_name(td.type), div, nexp, ntok);
  if (D % 128 || MID % 128) {
    fprintf(stderr, "K 不是 128 的倍数\n");
    return 2;
  }

  std::mt19937 rng(1);
  std::normal_distribution<double> nd(0.0, 1.0);

  double w_num = 0, w_den = 0;
  double up_we = 0, up_ae = 0, up_be = 0;  // 权重only / 激活only / 两者 的 hid rel-L2 累计
  double dn_we = 0, dn_ae = 0, dn_be = 0, dn_proto = 0;
  int cnt = 0;
  for (int e = 0; e < nexp; ++e) {
    const int ex = (int)(((unsigned)e * 7919 + 13) % E);
    Tensor2 G, U, DN;
    load_tensor(tg, ex, MID, D, div, G, "gate");
    load_tensor(tu, ex, MID, D, div, U, "up");
    load_tensor(td, ex, D, MID, div, DN, "down");
    for (int t = 0; t < ntok; ++t) {
      std::vector<double> x(D), xq(D);
      for (auto& v : x) v = nd(rng);
      quant_a4(x.data(), xq.data(), D);
      // up: hid = up * silu(gate)
      auto hid_of = [&](bool linw, bool linx) {
        const Tensor2 &g = G, &u = U;
        std::vector<double> gv(MID), uv(MID), hid(MID);
        if (linw) {
          matvec_lin(g, linx ? xq.data() : x.data(), gv.data());
          matvec_lin(u, linx ? xq.data() : x.data(), uv.data());
        } else {
          matvec(g, linx ? xq.data() : x.data(), gv.data());
          matvec(u, linx ? xq.data() : x.data(), uv.data());
        }
        for (int r = 0; r < MID; ++r) hid[r] = uv[r] * gv[r] / (1.0 + exp(-gv[r]));
        return hid;
      };
      const std::vector<double> h_ref = hid_of(false, false);
      up_we += rel_l2(hid_of(true, false), h_ref);
      up_ae += rel_l2(hid_of(false, true), h_ref);
      const std::vector<double> h_a4 = hid_of(true, true);
      up_be += rel_l2(h_a4, h_ref);
      // down：精确 hid 输入（隔离 down 自身误差）+ proto 复合情形（A4 hid 再量化）
      auto dn_of = [&](bool linw, const std::vector<double>& h, bool linx) {
        std::vector<double> hh = h, hq(MID);
        const double* inp = hh.data();
        if (linx) {
          quant_a4(hh.data(), hq.data(), MID);
          inp = hq.data();
        }
        std::vector<double> y(D);
        if (linw) matvec_lin(DN, inp, y.data());
        else matvec(DN, inp, y.data());
        return y;
      };
      const std::vector<double> y_ref = dn_of(false, h_ref, false);
      dn_we += rel_l2(dn_of(true, h_ref, false), y_ref);
      dn_ae += rel_l2(dn_of(false, h_ref, true), y_ref);
      dn_be += rel_l2(dn_of(true, h_ref, true), y_ref);
      // proto 口径：A4 的 down 输入是 A4 up 的输出，参考是 F16 链路的 pairs
      dn_proto += rel_l2(dn_of(true, h_a4, true), y_ref);
      cnt++;
    }
  }
  printf("up   hid rel-L2: 仅权重 %.4f | 仅激活 %.4f | 两者 %.4f  (proto 测得 0.2124)\n",
         up_we / cnt, up_ae / cnt, up_be / cnt);
  printf("down rel-L2: 仅权重 %.4f | 仅激活 %.4f | 两者 %.4f | 复合(proto口径) %.4f  (proto 测得 0.2989)\n",
         dn_we / cnt, dn_ae / cnt, dn_be / cnt, dn_proto / cnt);
  (void)w_num;
  (void)w_den;
  return 0;
}
