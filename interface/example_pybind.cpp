#include <torch/extension.h>

#include "example.h"

// Torch-facing wrapper: validate inputs, pull out raw data pointers, and hand
// them to the pure-C++ kernel in libmbs_kernel.so. All torch/ATen knowledge
// lives here, not in the kernel.
torch::Tensor tensor_add_tensor(torch::Tensor a, torch::Tensor b)
{
    TORCH_CHECK(a.sizes() == b.sizes(), "tensor_add: input shapes must match, got ",
                a.sizes(), " and ", b.sizes());
    TORCH_CHECK(a.scalar_type() == b.scalar_type(),
                "tensor_add: input dtypes must match, got ", a.scalar_type(),
                " and ", b.scalar_type());

    auto ac  = a.contiguous();
    auto bc  = b.contiguous();
    auto out = torch::empty_like(ac);

    const int64_t n = ac.numel();

    switch (ac.scalar_type())
    {
    case at::ScalarType::Float:
        tensor_add<float>(ac.data_ptr<float>(), bc.data_ptr<float>(),
                          out.data_ptr<float>(), n);
        break;
    case at::ScalarType::Double:
        tensor_add<double>(ac.data_ptr<double>(), bc.data_ptr<double>(),
                           out.data_ptr<double>(), n);
        break;
    default:
        TORCH_CHECK(false, "tensor_add: unsupported dtype ", ac.scalar_type());
    }

    return out;
}

void test_read_pattern(torch::Tensor& x, torch::Tensor& y) {
    read_pattern(x.data_ptr<uint8_t>(), x.stride(0), y.data_ptr<uint8_t>());
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.doc() = "mbs tensor add example kernel";
    m.def("tensor_add", &tensor_add_tensor, "Element-wise tensor addition (out = a + b)",
          pybind11::arg("a"), pybind11::arg("b"));
    m.def("test_read_pattern", &test_read_pattern, "",
          pybind11::arg("x"), pybind11::arg("y"));
    m.def("mbs_factor_experiment", &mbs_factor_experiment, "");
}
