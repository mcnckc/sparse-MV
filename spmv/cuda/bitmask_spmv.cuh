#include <cuda_fp16.h>

__global__ void bitmask_spmv(
    const __half *__restrict__ M_values, const unsigned char *__restrict__ M_mask, const int *__restrict__ M_row_indices,
    const __half *__restrict__ V,
    const int M_rows, const int M_cols, const float alpha, const float beta,
    __half *__restrict__ C)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M_rows) return;

    int val_start = M_row_indices[row];
    int val_end   = M_row_indices[row + 1];

    int bytes_per_row = (M_cols + 7) / 8;
    const unsigned char *mask_row = M_mask + row * bytes_per_row;
    const __half *row_vals = M_values + val_start;

    float sum = 0.0f;
    int val_idx = 0;

    for (int b = 0; b < bytes_per_row; ++b) {
        unsigned char byte = mask_row[b];
        for (int bit = 0; bit < 8; ++bit) {
            int col = b * 8 + bit;
            if (col >= M_cols) break;

            if (byte & (1 << bit)) {
                float a = __half2float(row_vals[val_idx]);
                float b = __half2float(V[col]);
                sum += a * b;
                ++val_idx;
            }
        }
    }

    float old_C = __half2float(C[row]);
    C[row] = __float2half(alpha * sum + beta * old_C);
}