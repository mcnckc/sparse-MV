#include <torch/extension.h> 

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
convert_bitmask(torch::Tensor M) {
    TORCH_CHECK(M.dim() == 2, "M must be a 2D tensor");
    TORCH_CHECK(M.dtype() == torch::kHalf, "M must be of type torch.float16");
    TORCH_CHECK(M.device().is_cpu(), "M must be on CPU");
    TORCH_CHECK(M.is_contiguous(), "M must be contiguous");

    int M_rows = static_cast<int>(M.size(0));
    int M_cols = static_cast<int>(M.size(1));

    int *row_indices = nullptr;
    unsigned char *mask = nullptr;
    __half *values = nullptr;

    convert_bitmask_cpu(
        reinterpret_cast<const __half*>(M.data_ptr<at::Half>()),
        M_rows, M_cols,
        &row_indices, &mask, &values
    );

    size_t total_elements = static_cast<size_t>(M_rows) * M_cols;
    size_t mask_bytes = (total_elements + 7) / 8;

    int nnz = row_indices[M_rows];

    auto options_row = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
    auto options_mask = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
    auto options_val  = torch::TensorOptions().dtype(torch::kHalf).device(torch::kCPU);

    torch::Tensor row_tensor = torch::from_blob(
        row_indices, {M_rows + 1}, 
        [row_indices](void*) { delete[] row_indices; }, options_row);

    torch::Tensor mask_tensor = torch::from_blob(
        mask, {static_cast<int64_t>(mask_bytes)}, 
        [mask](void*) { delete[] mask; }, options_mask);

    torch::Tensor values_tensor = torch::from_blob(
        values, {nnz}, 
        [values](void*) { delete[] values; }, options_val);

    return std::make_tuple(row_tensor, mask_tensor, values_tensor);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("convert_bitmask", &convert_bitmask, "Convert dense FP16 matrix to bitmask format");
}