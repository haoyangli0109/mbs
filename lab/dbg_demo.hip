// A small program to practice rocgdb on (see debug.md).
//
//   build/dbg_demo       out[i] = in[i-1] + in[i] + in[i+1] inside each block of 256
//   build/dbg_demo bug   the load index is off by 2^30 elements: a GPU memory access fault
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(x)                                                                    \
    do {                                                                            \
        hipError_t e_ = (x);                                                        \
        if (e_ != hipSuccess) {                                                     \
            std::printf("%s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_)); \
            std::exit(1);                                                           \
        }                                                                           \
    } while (0)

__global__ void neighbour_sum(const float* in, float* out, int shift)
{
    __shared__ float tile[256];
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + tid;
    tile[tid] = in[gid + shift];
    __syncthreads();
    const float left = tile[(tid + 255) % 256];
    const float right = tile[(tid + 1) % 256];
    out[gid] = left + tile[tid] + right;
}

int main(int argc, char** argv)
{
    const int n = 1 << 20;
    const int shift = argc > 1 && !strcmp(argv[1], "bug") ? 1 << 30 : 0;
    std::vector<float> h(n);
    for (int i = 0; i < n; ++i)
        h[i] = float(i % 7);
    float *in, *out;
    CHECK(hipMalloc(&in, n * sizeof(float)));
    CHECK(hipMalloc(&out, n * sizeof(float)));
    CHECK(hipMemcpy(in, h.data(), n * sizeof(float), hipMemcpyHostToDevice));
    neighbour_sum<<<n / 256, 256>>>(in, out, shift);
    CHECK(hipDeviceSynchronize());
    CHECK(hipMemcpy(h.data(), out, n * sizeof(float), hipMemcpyDeviceToHost));
    std::printf("out[1] = %g (expect 3)\n", h[1]);
    return 0;
}
