#include <torch/extension.h>

#include <c10/hip/HIPStream.h>

#include "gemm_a4w4.h"

// Torch-facing wrapper for the MXFP4 A4W4 GEMM. Validates the inputs, allocates
// the (M-padded) bf16 output, pulls out raw device pointers + the current HIP
// stream, and hands them to the torch-free launcher in libmbs_gemm_kernel.so.
//
//   A       : [M, K/2]  uint8 (fp4x2), plain row-major
//   B       : [N, K/2]  uint8 (fp4x2), preshuffled (16x16)
//   A_scale : uint8 (e8m0), shuffled   -- flat buffer
//   B_scale : uint8 (e8m0), shuffled   -- flat buffer
//   A_factor: [M] (or [M, 1]) uint8, MBS factor per row of A
//   B_factor: [N] (or [N, 1]) uint8, MBS factor per row of B
// Returns out[M, N] bf16 = (dequant(A) @ dequant(B)^T) * f(A_factor) * f(B_factor)^T,
// f(b) = 1 / (1 + b/256).
torch::Tensor gemm_a4w4(torch::Tensor A, torch::Tensor B, torch::Tensor A_scale,
                        torch::Tensor B_scale, torch::Tensor A_factor, torch::Tensor B_factor)
{
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "gemm_a4w4: A, B must be on device");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "gemm_a4w4: A, B must be 2D");

    auto Au  = A.view(at::kByte).contiguous();
    auto Bu  = B.view(at::kByte).contiguous();
    auto Asc = A_scale.view(at::kByte).contiguous();
    auto Bsc = B_scale.view(at::kByte).contiguous();
    auto Afc = A_factor.view(at::kByte).contiguous();
    auto Bfc = B_factor.view(at::kByte).contiguous();

    const int M  = static_cast<int>(Au.size(0));
    const int Kp = static_cast<int>(Au.size(1));  // packed bytes = K/2
    const int K  = Kp * 2;
    const int N  = static_cast<int>(Bu.size(0));

    TORCH_CHECK(Bu.size(1) == Kp, "gemm_a4w4: A, B must share K/2, got ",
                Au.size(1), " vs ", Bu.size(1));
    TORCH_CHECK(K % 32 == 0, "gemm_a4w4: K must be a multiple of 32, got ", K);
    TORCH_CHECK(N % 16 == 0, "gemm_a4w4: N must be a multiple of 16, got ", N);
    TORCH_CHECK(Afc.is_cuda() && Afc.numel() == M, "gemm_a4w4: A_factor must hold M = ", M,
                " bytes on device, got ", Afc.numel());
    TORCH_CHECK(Bfc.is_cuda() && Bfc.numel() == N, "gemm_a4w4: B_factor must hold N = ", N,
                " bytes on device, got ", Bfc.numel());

    const int Mpad       = (M + 31) / 32 * 32;
    const int scaleNv    = (K + 31) / 32;
    const int scaleN_pad = (scaleNv + 7) / 8 * 8;

    auto out = torch::empty(
        {Mpad, N},
        torch::TensorOptions().dtype(torch::kBFloat16).device(A.device()));

    hipStream_t stream = c10::cuda::getCurrentCUDAStream(A.device().index());

    const bool launched = gemm_a4w4_mxfp4_launch(
        Au.data_ptr<uint8_t>(), Bu.data_ptr<uint8_t>(), Asc.data_ptr<uint8_t>(),
        Bsc.data_ptr<uint8_t>(), Afc.data_ptr<uint8_t>(), Bfc.data_ptr<uint8_t>(),
        out.data_ptr(), M, N, K, Mpad, scaleN_pad, reinterpret_cast<void*>(stream));
    TORCH_CHECK(launched, "gemm_a4w4: MBS factors are only supported by the workgroup kernel "
                "(K % 1024 == 0 and enough output tiles); got M=", M, " N=", N, " K=", K);

    return out.narrow(0, 0, M);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.doc() = "mbs MXFP4 A4W4 GEMM (gfx950)";
    m.def("gemm_a4w4", &gemm_a4w4,
          "MXFP4 A4W4 GEMM with MBS factors: "
          "out[M,N] = dequant(A) @ dequant(B)^T * f(A_factor) * f(B_factor)^T",
          pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("A_scale"),
          pybind11::arg("B_scale"), pybind11::arg("A_factor"), pybind11::arg("B_factor"));
}
