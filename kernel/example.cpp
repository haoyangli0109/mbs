#include "example.h"
#include <iostream>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include <type_traits>
#include <utility>
#include <algorithm>
#include <string>
#include <unordered_map>

// // Plain loop over raw buffers -- no torch, no ATen.
template <typename T>
void tensor_add(const T* a, const T* b, T* out, int64_t n)
{
    for (int64_t i = 0; i < n; ++i)
    {
        out[i] = a[i] + b[i];
    }
    std::cout << "hahah: in tensor_add" << std::endl;
}

// Explicit instantiations so the kernel library exports concrete symbols.
template void tensor_add<float>(const float*, const float*, float*, int64_t);
template void tensor_add<double>(const double*, const double*, double*, int64_t);

// __global__ void read_pattern_kernel1(const uint8_t* x, size_t x_stride, uint8_t* y) {
//     uint8_t buffer[64];
//     size_t offset = blockIdx.x * x_stride + threadIdx.x * 64;
//     #pragma unroll
//     for (int i = 0; i < 64; i++) buffer[i] = x[offset + i];
//     #pragma unroll
//     for (int i = 0; i < 64; i++) y[offset + i] = buffer[i];
// }

__global__ void read_pattern_kernel2(const uint8_t* x, size_t x_stride, uint8_t* y) {
    uint8_t buffer[64];
    size_t offset = blockIdx.x * x_stride + threadIdx.x * 16;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 16; j++)
            buffer[i * 16 + j] = x[offset + j];
        offset += 64 * 16;
    }
    offset = blockIdx.x * x_stride + threadIdx.x * 16;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 16; j++)
            y[offset + j] = buffer[i * 16 + j];
        offset += 64 * 16;
    }
}

#ifndef MBS_ASM_DIR
#define MBS_ASM_DIR "build/asm"
#endif

#define HIP_CHECK(cmd)                                                              \
    do {                                                                            \
        hipError_t e_ = (cmd);                                                      \
        if (e_ != hipSuccess) {                                                     \
            std::cerr << #cmd << " failed: " << hipGetErrorString(e_) << std::endl; \
            std::abort();                                                           \
        }                                                                           \
    } while (0)

// Load the asm build of a read_pattern kernel (build/asm/<kernel_name>.hsaco)
// once per name and cache the function handle.
static hipFunction_t get_read_pattern_asm_kernel(const std::string& kernel_name) {
    static std::unordered_map<std::string, hipFunction_t> cache;
    auto it = cache.find(kernel_name);
    if (it != cache.end()) return it->second;

    const std::string path = std::string(MBS_ASM_DIR) + "/" + kernel_name + ".hsaco";
    // Itanium mangling of `void <name>(const uint8_t*, size_t, uint8_t*)`.
    const std::string symbol =
        "_Z" + std::to_string(kernel_name.size()) + kernel_name + "PKhmPh";

    hipModule_t mod;
    HIP_CHECK(hipModuleLoad(&mod, path.c_str()));
    hipFunction_t f;
    HIP_CHECK(hipModuleGetFunction(&f, mod, symbol.c_str()));
    cache.emplace(kernel_name, f);
    return f;
}

void read_pattern(const uint8_t* x, size_t x_stride, uint8_t* y) {
    std::cout << "x_stride = " << x_stride << std::endl;
    // for (int i = 0; i < 10; i++) read_pattern_kernel1<<<256, 64, 0, 0>>>(x, x_stride, y);
    // for (int i = 0; i < 10; i++) read_pattern_kernel2<<<256, 64, 0, 0>>>(x, x_stride, y);
    // Same signature as read_pattern_kernel1: (const uint8_t*, size_t, uint8_t*).
    void* args[] = {&x, &x_stride, &y};
    auto fn1 = get_read_pattern_asm_kernel("read_pattern_kernel1");
    auto fn2 = get_read_pattern_asm_kernel("read_pattern_kernel2");
    for (int i = 0; i < 10; i++)
        HIP_CHECK(hipModuleLaunchKernel(fn1,
                                        256, 1, 1,  // grid
                                        64, 1, 1,   // block
                                        0, 0, args, nullptr));
    for (int i = 0; i < 10; i++)
        HIP_CHECK(hipModuleLaunchKernel(fn2,
                                        256, 1, 1,  // grid
                                        64, 1, 1,   // block
                                        0, 0, args, nullptr));
}

typedef float f32x4 __attribute__((ext_vector_type(4)));

template<int MSUB, int NSUB>
__global__ void mbs_factor_experiment_kernel_v1(
    uint8_t * M_factor_ptr,
    uint8_t * N_factor_ptr,
    float * out
) {
    f32x4 acc[MSUB][NSUB];
    #pragma unroll
    for (int mi = 0; mi < MSUB; ++mi)
        #pragma unroll
        for (int ni = 0; ni < NSUB; ++ni)
            acc[mi][ni] = f32x4{1.f, 1.f, 1.f, 1.f};

    constexpr int M_factor_per_thread = (16 * MSUB + 63) / 64;
    constexpr int N_factor_per_thread = (16 * NSUB + 63) / 64;
    uint8_t M_factor_buffer[M_factor_per_thread];
    uint8_t N_factor_buffer[N_factor_per_thread];

    __shared__ float M_factor_lds[M_factor_per_thread * 64];
    __shared__ float N_factor_lds[4][N_factor_per_thread * 64];

    int lane = threadIdx.x % 64;
    int wave = threadIdx.x / 64;
    #pragma unroll
    for (int i = 0; i < N_factor_per_thread; i++)
        N_factor_buffer[i] = N_factor_ptr[1024 * blockIdx.x + lane * N_factor_per_thread + i];

    if (wave == 0) {
        #pragma unroll
        for (int i = 0; i < M_factor_per_thread; i++)
            M_factor_buffer[i] = M_factor_ptr[1024 * blockIdx.x + lane * M_factor_per_thread + i];
    }

    auto convert_byte_to_float = [&](uint8_t b) {
        float one = 1.0;
        uint32_t* bits_one = (uint32_t*)(&one);
        *bits_one |= (b << 15);
        return 1.0 / one;
    };

    #pragma unroll
    for (int i = 0; i < N_factor_per_thread; i++)
        N_factor_lds[wave][lane * N_factor_per_thread + i] = convert_byte_to_float(N_factor_buffer[i]);

    if (wave == 0) {
        #pragma unroll
        for (int i = 0; i < M_factor_per_thread; i++)
            M_factor_lds[lane * M_factor_per_thread + i] = convert_byte_to_float(M_factor_buffer[i]);
    }
    
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < MSUB; i++) {
        #pragma unroll
        for (int j = 0; j < NSUB; j++) {
            int row = 16 * i + lane % 16;
            int col = 16 * j + lane / 4;  // lane / 16 * 4
            float M_factor = M_factor_lds[row];
            f32x4 N_factor = f32x4{
                N_factor_lds[wave][col],
                N_factor_lds[wave][col + 1],
                N_factor_lds[wave][col + 2],
                N_factor_lds[wave][col + 3]
            };
            #pragma unroll
            for (int k = 0; k < 4; k++)
                acc[i][j][k] *= (M_factor * N_factor[k]);
            *(reinterpret_cast<f32x4*>(out + 4 * lane)) = acc[i][j];
        }
    }
}

template<int MSUB, int NSUB>
__global__ void mbs_factor_experiment_kernel_v2(
    uint8_t * M_factor_ptr,
    uint8_t * N_factor_ptr,
    float * out
) {
    f32x4 acc[MSUB][NSUB];
    #pragma unroll
    for (int mi = 0; mi < MSUB; ++mi)
        #pragma unroll
        for (int ni = 0; ni < NSUB; ++ni)
            acc[mi][ni] = f32x4{1.f, 1.f, 1.f, 1.f};

    constexpr int M_factor_per_thread = (16 * MSUB + 255) / 256;  // M_factors are dealt with by the whole block
    constexpr int N_factor_per_thread = (16 * NSUB + 63) / 64;
    uint8_t M_factor_buffer[M_factor_per_thread];
    uint8_t N_factor_buffer[N_factor_per_thread];

    __shared__ float M_factor_lds[M_factor_per_thread * 256];
    __shared__ float N_factor_lds[4][N_factor_per_thread * 64];

    int lane = threadIdx.x % 64;
    int wave = threadIdx.x / 64;
    #pragma unroll
    for (int i = 0; i < N_factor_per_thread; i++)
        N_factor_buffer[i] = N_factor_ptr[1024 * blockIdx.x + wave * 256 + lane * N_factor_per_thread + i];

    #pragma unroll
    for (int i = 0; i < M_factor_per_thread; i++)
        M_factor_buffer[i] = M_factor_ptr[1024 * blockIdx.x + threadIdx.x * M_factor_per_thread + i];

    auto convert_byte_to_float = [&](uint8_t b) {
        float one = 1.0;
        uint32_t* bits_one = (uint32_t*)(&one);
        *bits_one |= (b << 15);
        return 1.0 / one;
    };

    #pragma unroll
    for (int i = 0; i < N_factor_per_thread; i++)
        N_factor_lds[wave][lane * N_factor_per_thread + i] = convert_byte_to_float(N_factor_buffer[i]);

    #pragma unroll
    for (int i = 0; i < M_factor_per_thread; i++)
        M_factor_lds[threadIdx.x * M_factor_per_thread + i] = convert_byte_to_float(M_factor_buffer[i]);
    
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < MSUB; i++) {
        #pragma unroll
        for (int j = 0; j < NSUB; j++) {
            int row = 16 * i + lane % 16;
            int col = 16 * j + lane / 4;  // lane / 16 * 4
            float M_factor = M_factor_lds[row];
            f32x4 N_factor = f32x4{
                N_factor_lds[wave][col],
                N_factor_lds[wave][col + 1],
                N_factor_lds[wave][col + 2],
                N_factor_lds[wave][col + 3]
            };
            #pragma unroll
            for (int k = 0; k < 4; k++)
                acc[i][j][k] *= (M_factor * N_factor[k]);
            *(reinterpret_cast<f32x4*>(out + 4 * lane)) = acc[i][j];
        }
    }
}

template<int MSUB, int NSUB>
__global__ void mbs_factor_experiment_kernel_v3(
    uint8_t * M_factor_ptr,
    uint8_t * N_factor_ptr,
    float * out
) {
    f32x4 acc[MSUB][NSUB];
    #pragma unroll
    for (int mi = 0; mi < MSUB; ++mi)
        #pragma unroll
        for (int ni = 0; ni < NSUB; ++ni)
            acc[mi][ni] = f32x4{1.f, 1.f, 1.f, 1.f};

    auto convert_byte_to_float = [&](uint8_t b) {
        float one = 1.0;
        uint32_t* bits_one = (uint32_t*)(&one);
        *bits_one |= (b << 15);
        return 1.0 / one;
    };

    constexpr int M_factor_per_thread = (16 * MSUB + 63) / 64;
    constexpr int N_factor_per_thread = (16 * NSUB + 63) / 64;
    __shared__ float M_factor_lds[M_factor_per_thread * 64];
    __shared__ float N_factor_lds[4][N_factor_per_thread * 64];

    int lane = threadIdx.x % 64;
    int wave = threadIdx.x / 64;
    if (wave == 0) {
        uint8_t M_factor_buffer[M_factor_per_thread];
        uint8_t N_factor_buffer[N_factor_per_thread];
        #pragma unroll
        for (int i = 0; i < M_factor_per_thread; i++)
            M_factor_buffer[i] = M_factor_ptr[1024 * blockIdx.x + lane * M_factor_per_thread + i];
        
        #pragma unroll
        for (int i = 0; i < N_factor_per_thread; i++)
            N_factor_buffer[i] = N_factor_ptr[1024 * blockIdx.x + lane * N_factor_per_thread + i];

        #pragma unroll
        for (int i = 0; i < M_factor_per_thread; i++)
            M_factor_lds[lane * M_factor_per_thread + i] = convert_byte_to_float(M_factor_buffer[i]);
        
        __syncthreads();
        __builtin_amdgcn_sched_barrier(0);
        #pragma unroll
        for (int i = 0; i < N_factor_per_thread; i++)
            N_factor_lds[wave][lane * N_factor_per_thread + i] = convert_byte_to_float(N_factor_buffer[i]);

        #pragma unroll
        for (int i = 0; i < MSUB; i++) {
            #pragma unroll
            for (int j = 0; j < NSUB; j++) {
                int row = 16 * i + lane % 16;
                int col = 16 * j + lane / 4;  // lane / 16 * 4
                float M_factor = M_factor_lds[row];
                f32x4 N_factor = f32x4{
                    N_factor_lds[wave][col],
                    N_factor_lds[wave][col + 1],
                    N_factor_lds[wave][col + 2],
                    N_factor_lds[wave][col + 3]
                };
                #pragma unroll
                for (int k = 0; k < 4; k++)
                    acc[i][j][k] *= (M_factor * N_factor[k]);
                    *(reinterpret_cast<f32x4*>(out + 4 * lane)) = acc[i][j];
            }
        }
    } else {
        uint8_t N_factor_buffer[N_factor_per_thread];

        #pragma unroll
        for (int i = 0; i < N_factor_per_thread; i++)
            N_factor_buffer[i] = N_factor_ptr[1024 * blockIdx.x + lane * N_factor_per_thread + i];

        #pragma unroll
        for (int i = 0; i < N_factor_per_thread; i++)
            N_factor_lds[wave][lane * N_factor_per_thread + i] = convert_byte_to_float(N_factor_buffer[i]);
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < MSUB; i++) {
            #pragma unroll
            for (int j = 0; j < NSUB; j++) {
                int row = 16 * i + lane % 16;
                int col = 16 * j + lane / 4;  // lane / 16 * 4
                float M_factor = M_factor_lds[row];
                f32x4 N_factor = f32x4{
                    N_factor_lds[wave][col],
                    N_factor_lds[wave][col + 1],
                    N_factor_lds[wave][col + 2],
                    N_factor_lds[wave][col + 3]
                };
                #pragma unroll
                for (int k = 0; k < 4; k++)
                    acc[i][j][k] *= (M_factor * N_factor[k]);
                    *(reinterpret_cast<f32x4*>(out + 4 * lane)) = acc[i][j];
            }
        }
    }
}

template<int MSUB, int NSUB>
__global__ void mbs_factor_experiment_kernel_v4(
    uint8_t * M_factor_ptr,
    uint8_t * N_factor_ptr,
    float * out
) {
    f32x4 acc[MSUB][NSUB];
    #pragma unroll
    for (int mi = 0; mi < MSUB; ++mi)
        #pragma unroll
        for (int ni = 0; ni < NSUB; ++ni)
            acc[mi][ni] = f32x4{1.f, 1.f, 1.f, 1.f};

    auto convert_byte_to_float = [&](uint8_t b) {
        float one = 1.0;
        uint32_t* bits_one = (uint32_t*)(&one);
        *bits_one |= (b << 15);
        return 1.0 / one;
    };

    constexpr int M_factor_per_thread = (16 * MSUB + 63) / 64;
    constexpr int N_factor_per_thread = (16 * NSUB + 63) / 64;
    __shared__ float M_factor_lds[M_factor_per_thread * 64];
    __shared__ float N_factor_lds[4][N_factor_per_thread * 64];

    int lane = threadIdx.x % 64;
    int wave = threadIdx.x / 64;
    if (wave == 0) {
        uint8_t M_factor_buffer[M_factor_per_thread];
        uint8_t N_factor_buffer[N_factor_per_thread];
        #pragma unroll
        for (int i = 0; i < M_factor_per_thread; i++)
            M_factor_buffer[i] = M_factor_ptr[1024 * blockIdx.x + lane * M_factor_per_thread + i];
        
        #pragma unroll
        for (int i = 0; i < N_factor_per_thread; i++)
            N_factor_buffer[i] = N_factor_ptr[1024 * blockIdx.x + lane * N_factor_per_thread + i];

        #pragma unroll
        for (int i = 0; i < M_factor_per_thread; i++)
            M_factor_lds[lane * M_factor_per_thread + i] = convert_byte_to_float(M_factor_buffer[i]);
        
        #pragma unroll
        for (int i = 0; i < N_factor_per_thread; i++)
            N_factor_lds[wave][lane * N_factor_per_thread + i] = convert_byte_to_float(N_factor_buffer[i]);

        #pragma unroll
        for (int i = 0; i < MSUB; i++) {
            #pragma unroll
            for (int j = 0; j < NSUB; j++) {
                int row = 16 * i + lane % 16;
                int col = 16 * j + lane / 4;  // lane / 16 * 4
                float M_factor = M_factor_lds[row];
                f32x4 N_factor = f32x4{
                    N_factor_lds[wave][col],
                    N_factor_lds[wave][col + 1],
                    N_factor_lds[wave][col + 2],
                    N_factor_lds[wave][col + 3]
                };
                #pragma unroll
                for (int k = 0; k < 4; k++)
                    acc[i][j][k] *= (M_factor * N_factor[k]);
                    *(reinterpret_cast<f32x4*>(out + 4 * lane)) = acc[i][j];
            }
        }
    } else {
        return;
    }
}

void mbs_factor_experiment() {
    uint8_t * M_factor;
    uint8_t * N_factor;
    float * out;
    HIP_CHECK(hipMalloc(&M_factor, 1024 * 256));
    HIP_CHECK(hipMalloc(&N_factor, 1024 * 256));
    HIP_CHECK(hipMalloc(&out, 16 * 16 * 1024 * sizeof(float)));
    int run_times = 20;
    for (int i = 0; i < run_times; i++)
        mbs_factor_experiment_kernel_v1<10, 4><<<256, 64 * 4, 0, 0>>>(
            M_factor, N_factor, out
        );
    for (int i = 0; i < run_times; i++)
        mbs_factor_experiment_kernel_v2<10, 4><<<256, 64 * 4, 0, 0>>>(
            M_factor, N_factor, out
        );
    for (int i = 0; i < run_times; i++)
        mbs_factor_experiment_kernel_v3<10, 4><<<256, 64 * 4, 0, 0>>>(
            M_factor, N_factor, out
        );
    // for (int i = 0; i < run_times; i++)
    //     mbs_factor_experiment_kernel_v4<10, 4><<<256, 64 * 4, 0, 0>>>(
    //         M_factor, N_factor, out
    //     );
}
