#include <cuda_fp16.h>
#include <cstddef>
#include <algorithm>

void convert_bitmask_cpu(
    const __half *M, int M_rows, int M_cols,
    int **row_indices, unsigned char **mask, __half **values) {
    size_t total_elements = static_cast<size_t>(M_rows) * static_cast<size_t>(M_cols);

    size_t nnz = 0;
    for (size_t i = 0; i < total_elements; ++i) {
        if (__half2float(M[i]) != 0.0f)
            ++nnz;
    }

    *row_indices = new int[M_rows + 1];

    *values = new __half[nnz];

    size_t mask_bytes = (total_elements + 7) / 8;
    *mask = new unsigned char[mask_bytes]();

    size_t val_idx = 0; 
    int *row_ptr = *row_indices;
    unsigned char *mask_ptr = *mask;

    for (int r = 0; r < M_rows; ++r) {
        row_ptr[r] = static_cast<int>(val_idx);
        for (int c = 0; c < M_cols; ++c) {
            size_t idx = static_cast<size_t>(r) * M_cols + c;
            __half h = M[idx];
            if (__half2float(h) != 0.0f) {
                (*values)[val_idx++] = h;
                size_t byte_idx = idx >> 3;
                size_t bit_pos  = idx & 7;
                mask_ptr[byte_idx] |= (1u << bit_pos);
            }
        }
    }
    row_ptr[M_rows] = static_cast<int>(nnz);
}