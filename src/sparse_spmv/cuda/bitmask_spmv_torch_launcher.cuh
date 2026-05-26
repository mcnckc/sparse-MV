#include <torch/extension.h>
#include <cuda.h>
#include <cuda_fp16.h>

torch::Tensor bitmask_spmv_launch(
    torch::Tensor M_values,
    torch::Tensor M_mask,
    torch::Tensor M_row_indices,
    int M_rows,
    int M_cols,
    torch::Tensor V,
    float alpha,
    float beta)
{
    // Input validations
    TORCH_CHECK(M_values.is_cuda(), "M_values must be a CUDA tensor");
    TORCH_CHECK(M_mask.is_cuda(),    "M_mask must be a CUDA tensor");
    TORCH_CHECK(M_row_indices.is_cuda(), "M_row_indices must be a CUDA tensor");
    TORCH_CHECK(V.is_cuda(),         "V must be a CUDA tensor");

    TORCH_CHECK(M_values.dtype() == torch::kFloat16, "M_values must be float16");
    TORCH_CHECK(M_mask.dtype() == torch::kUInt8,     "M_mask must be uint8");
    TORCH_CHECK(M_row_indices.dtype() == torch::kInt32, "M_row_indices must be int32");
    TORCH_CHECK(V.dtype() == torch::kFloat16,         "V must be float16");

    TORCH_CHECK(M_values.is_contiguous(), "M_values must be contiguous");
    TORCH_CHECK(M_mask.is_contiguous(),   "M_mask must be contiguous");
    TORCH_CHECK(M_row_indices.is_contiguous(), "M_row_indices must be contiguous");
    TORCH_CHECK(V.is_contiguous(),        "V must be contiguous");

    // Create output tensor C (zero‑filled)
    auto C = torch::zeros({M_rows}, torch::TensorOptions().dtype(torch::kFloat16).device(M_values.device()));

    // Grid/block setup
    const int threads = 256;
    const int blocks = (M_rows + threads - 1) / threads;

    bitmask_spmv<<<blocks, threads>>>(
        reinterpret_cast<__half*>(M_values.data_ptr()),
        M_mask.data_ptr<unsigned char>(),
        M_row_indices.data_ptr<int>(),
        reinterpret_cast<__half*>(V.data_ptr()),
        M_rows,
        M_cols,
        alpha,
        beta,
        reinterpret_cast<__half*>(C.data_ptr())
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel launch failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bitmask_spmv_launch", &bitmask_spmv_launch, "Bitmask SpMV (CUDA)");
}
