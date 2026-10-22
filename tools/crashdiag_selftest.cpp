// crashdiag_selftest — src/crashdiag.h 的独立测试程序（不依赖 HIP）。
//   crashdiag_selftest <kind>   安装处理器、写几条飞行记录后按 kind 崩溃
//   crashdiag_selftest env      只打印环境快照
// kind 见 crashdiag::self_test_kinds()。由 tools/crashdiag_test.sh 调用。
#include "../src/crashdiag.h"

int main(int argc, char** argv) {
  crashdiag::install("selftest", "test (crashdiag_selftest)");
  crashdiag::append_env(crashdiag::env_report());
  if (argc < 2) {
    fprintf(stderr, "usage: %s <%s|env>\n", argv[0], crashdiag::self_test_kinds());
    return 2;
  }
  if (!strcmp(argv[1], "env")) {
    fputs(crashdiag::g.env, stdout);
    return 0;
  }
  for (int i = 0; i < 3; i++) crashdiag::note("selftest note %d", i);
  crashdiag::site("tools/crashdiag_selftest.cpp:20 hipMemcpyAsync(d_pos, &pos, 4, H2D)");
  crashdiag::note("about to crash: %s", argv[1]);
  if (!crashdiag::self_test(argv[1])) {
    fprintf(stderr, "unknown kind %s (known: %s)\n", argv[1], crashdiag::self_test_kinds());
    return 2;
  }
  return 0;
}
