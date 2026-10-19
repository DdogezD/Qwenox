// gguf_expert_rows.cpp — dump dequantized routed-expert rows from a GGUF shard.
//
// Usage: gguf_expert_rows <shard1.gguf> <layer> <gate_up|down> <e0> <e1>
//
// Output (raw little-endian f32, no header):
//   gate_up: for each expert e in [e0,e1): 640 gate rows, then 640 up rows,
//            each of 2560 f32 => [e1-e0, 1280, 2560]
//   down:    for each expert e in [e0,e1): 2560 rows of 640 f32
//            => [e1-e0, 2560, 640]
//
// This matches the hgn v2 fused gate_up_proj / down_proj layout used by the
// i4r encoder. Dequantization is the engine's own gguf::dequant_row.
#include <cmath>
#include "gguf.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void die(const char* msg) {
  fprintf(stderr, "gguf_expert_rows: %s\n", msg);
  exit(1);
}

int main(int argc, char** argv) {
  if (argc != 6) {
    fprintf(stderr, "usage: %s <shard1.gguf> <layer> <gate_up|down> <e0> <e1>\n",
            argv[0]);
    return 2;
  }
  const std::string path = argv[1];
  const int layer = atoi(argv[2]);
  const std::string role = argv[3];
  const int e0 = atoi(argv[4]);
  const int e1 = atoi(argv[5]);
  if (layer < 0 || e0 < 0 || e1 <= e0) die("bad layer/expert range");
  if (role != "gate_up" && role != "down") die("role must be gate_up or down");

  gguf::File f(path);
  char name[96];
  const gguf::Tensor* tg = nullptr;
  const gguf::Tensor* tu = nullptr;
  const gguf::Tensor* td = nullptr;
  if (role == "gate_up") {
    snprintf(name, sizeof name, "blk.%d.ffn_gate_exps.weight", layer);
    tg = &f.at(name);
    snprintf(name, sizeof name, "blk.%d.ffn_up_exps.weight", layer);
    tu = &f.at(name);
  } else {
    snprintf(name, sizeof name, "blk.%d.ffn_down_exps.weight", layer);
    td = &f.at(name);
  }

  const gguf::Tensor* t0 = role == "gate_up" ? tg : td;
  const uint64_t K = t0->ne[0];
  const uint64_t R = t0->ne[1];
  const uint64_t E = t0->ne[2];
  if (role == "gate_up" && (tu->ne[0] != K || tu->ne[1] != R || tu->ne[2] != E))
    die("gate/up shape mismatch");
  if (role == "gate_up" && tu->type != tg->type)
    die("gate/up type mismatch");
  if (e1 > (int)E) die("expert range exceeds tensor");

  const uint64_t rb = t0->row_bytes();
  std::vector<float> row(K);
  std::vector<float> all;
  // Batch a few rows per fwrite to keep the pipe busy without a huge buffer.
  const size_t rows_per_flush = 32;
  std::vector<float> buf;
  buf.reserve(rows_per_flush * K);

  auto emit_row = [&](const gguf::Tensor* t, uint64_t e, uint64_t r) {
    const uint8_t* src = t->data + (e * R + r) * rb;
    gguf::dequant_row(t->type, src, row.data(), K);
    buf.insert(buf.end(), row.begin(), row.end());
    if (buf.size() >= rows_per_flush * K) {
      if (fwrite(buf.data(), sizeof(float), buf.size(), stdout) != buf.size())
        die("stdout write failed");
      buf.clear();
    }
  };

  for (int e = e0; e < e1; e++) {
    if (role == "gate_up") {
      for (uint64_t r = 0; r < R; r++) emit_row(tg, e, r);
      for (uint64_t r = 0; r < R; r++) emit_row(tu, e, r);
    } else {
      for (uint64_t r = 0; r < R; r++) emit_row(td, e, r);
    }
  }
  if (!buf.empty() &&
      fwrite(buf.data(), sizeof(float), buf.size(), stdout) != buf.size())
    die("stdout write failed");
  if (fflush(stdout) != 0) die("stdout flush failed");
  return 0;
}
