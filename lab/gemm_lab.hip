// GEMM lab: one MXFP4 GEMM problem, several kernel versions, one harness.
//
//   ./gemm_lab <version> [M N K] [--iters N] [--warmup-ms T] [--check S]
//
// Fills A, B and their scales with random data, runs the chosen version once and
// compares S sampled outputs with a CPU reference, then launches the kernel back to
// back for T ms (the GPU clock takes a few hundred ms to settle) and times N more
// launches. For profiler runs use --warmup-ms 0 --iters 2 --check 0.
#include "common.h"
#include "v0_naive.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define HIP_CHECK(x)                                                                  \
    do {                                                                              \
        hipError_t e_ = (x);                                                          \
        if (e_ != hipSuccess) {                                                       \
            std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x,             \
                         hipGetErrorString(e_));                                      \
            std::exit(1);                                                             \
        }                                                                             \
    } while (0)

struct Version {
    const char* name;
    void (*launch)(const Problem&, hipStream_t);
    int bm, bn, bk;  // the kernel needs M % bm == N % bn == K % bk == 0
};

static const Version kVersions[] = {
    {"v0", v0::launch, v0::BM, v0::BN, v0::BK},
};

static double fp4_value(int code)
{
    static const double mag[8] = {0, 0.5, 1, 1.5, 2, 3, 4, 6};
    return (code & 8 ? -1.0 : 1.0) * mag[code & 7];
}

static double bf16_to_double(__hip_bfloat16 v)
{
    uint16_t bits;
    std::memcpy(&bits, &v, 2);
    uint32_t f = uint32_t(bits) << 16;
    float x;
    std::memcpy(&x, &f, 4);
    return x;
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <version> [M N K] [--iters N] [--warmup-ms T] [--check S]\n",
                     argv[0]);
        return 1;
    }
    const Version* ver = nullptr;
    for (const Version& v : kVersions)
        if (std::string(argv[1]) == v.name)
            ver = &v;
    if (!ver) {
        std::fprintf(stderr, "unknown version %s\n", argv[1]);
        return 1;
    }
    int M = 8192, N = 8192, K = 5120, iters = 200, warmup_ms = 500, samples = 1024;
    int pos = 0;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--iters")
            iters = std::atoi(argv[++i]);
        else if (a == "--warmup-ms")
            warmup_ms = std::atoi(argv[++i]);
        else if (a == "--check")
            samples = std::atoi(argv[++i]);
        else
            (pos == 0 ? M : pos == 1 ? N : K) = std::atoi(argv[i]), ++pos;
    }
    if (M % ver->bm || N % ver->bn || K % ver->bk) {
        std::fprintf(stderr, "%s needs M %% %d == N %% %d == K %% %d == 0\n", ver->name, ver->bm,
                     ver->bn, ver->bk);
        return 1;
    }

    std::mt19937 rng(1234);
    std::vector<uint8_t> A(size_t(M) * K / 2), B(size_t(N) * K / 2);
    std::vector<uint8_t> As(size_t(M) * K / 32), Bs(size_t(N) * K / 32);
    for (auto& x : A) x = rng() & 0xff;
    for (auto& x : B) x = rng() & 0xff;
    for (auto& x : As) x = 120 + rng() % 15;
    for (auto& x : Bs) x = 120 + rng() % 15;

    uint8_t *dA, *dB, *dAs, *dBs;
    __hip_bfloat16* dC;
    HIP_CHECK(hipMalloc(&dA, A.size()));
    HIP_CHECK(hipMalloc(&dB, B.size()));
    HIP_CHECK(hipMalloc(&dAs, As.size()));
    HIP_CHECK(hipMalloc(&dBs, Bs.size()));
    HIP_CHECK(hipMalloc(&dC, size_t(M) * N * 2));
    HIP_CHECK(hipMemcpy(dA, A.data(), A.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, B.data(), B.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dAs, As.data(), As.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dBs, Bs.data(), Bs.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(dC, 0xff, size_t(M) * N * 2));
    const Problem p{M, N, K, dA, dB, dAs, dBs, dC};

    ver->launch(p, nullptr);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    if (samples > 0) {
        std::vector<__hip_bfloat16> C(size_t(M) * N);
        HIP_CHECK(hipMemcpy(C.data(), dC, C.size() * 2, hipMemcpyDeviceToHost));
        double worst = 0;
        for (int s = 0; s < samples; ++s) {
            const int m = rng() % M, n = rng() % N;
            double ref = 0, norm = 0;
            for (int kb = 0; kb < K / 32; ++kb) {
                double blk = 0;
                for (int j = 0; j < 16; ++j) {
                    const int a = A[size_t(m) * K / 2 + kb * 16 + j];
                    const int b = B[size_t(n) * K / 2 + kb * 16 + j];
                    blk += fp4_value(a & 15) * fp4_value(b & 15) + fp4_value(a >> 4) * fp4_value(b >> 4);
                }
                const double sc = std::ldexp(1.0, As[size_t(m) * K / 32 + kb] + Bs[size_t(n) * K / 32 + kb] - 254);
                ref += blk * sc;
                norm += blk * sc * blk * sc;
            }
            const double err = std::fabs(bf16_to_double(C[size_t(m) * N + n]) - ref) / (std::sqrt(norm) + 1e-30);
            worst = std::isnan(err) ? INFINITY : std::fmax(worst, err);
        }
        std::printf("check   %d samples, max error %.2e of the block-sum norm: %s\n", samples, worst,
                    worst < 3e-2 ? "PASS" : "FAIL");
    }

    const auto t0 = std::chrono::steady_clock::now();
    while (std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() < warmup_ms) {
        for (int i = 0; i < 10; ++i)
            ver->launch(p, nullptr);
        HIP_CHECK(hipDeviceSynchronize());
    }
    hipEvent_t e0, e1;
    HIP_CHECK(hipEventCreate(&e0));
    HIP_CHECK(hipEventCreate(&e1));
    HIP_CHECK(hipEventRecord(e0));
    for (int i = 0; i < iters; ++i)
        ver->launch(p, nullptr);
    HIP_CHECK(hipEventRecord(e1));
    HIP_CHECK(hipEventSynchronize(e1));
    float ms = 0;
    HIP_CHECK(hipEventElapsedTime(&ms, e0, e1));
    const double us = 1e3 * ms / iters;
    std::printf("%-4s M=%d N=%d K=%d  %9.1f us  %7.1f TFLOPS  (%d launches after %d ms warm-up)\n",
                ver->name, M, N, K, us, 2.0 * M * N * K / us / 1e6, iters, warmup_ms);

    HIP_CHECK(hipFree(dA));
    HIP_CHECK(hipFree(dB));
    HIP_CHECK(hipFree(dAs));
    HIP_CHECK(hipFree(dBs));
    HIP_CHECK(hipFree(dC));
    return 0;
}
