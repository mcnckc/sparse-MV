from torch.utils.cpp_extension import load_inline
import os
import torch
from importlib import resources
from importlib.metadata import version


_cuda_dir = "cuda"
_extension_convert_bitmask = load_inline(
    name='convert_bitmask_cpu_ext',
    sources=[os.path.join(_cuda_dir,  'convert_bitmask_torch_launcher.cuh')],
    extra_include_paths=[_cuda_dir],
    verbose=True,
)

def convert_bitmask(M):
    assert M.is_contiguous()
    if M.device.type == "cuda":
        M_bitmask = _extension_convert_bitmask.convert_bitmask(M.to("cpu"))
        return move_to_device(M_bitmask, "cuda")

    elif M.device.type == "cpu":
        M_bitmask = _extension_convert_bitmask.convert_bitmask(M)
        return M_bitmask
    else:
        raise NotImplementedError()


_bitmask_spmv_module = load_inline(
    name='bitmask_spmv_cuda_ext',
    cpp_sources='',    
    cuda_sources=[os.path.join(_cuda_dir,  'bitmask_spmv_torch_launcher.cuh'), os.path.join(_cuda_dir,  'bitmask_spmv.cuh')],
    functions=['bitmask_spmv_launch'],
    extra_cuda_cflags=['-O3'],
    verbose=False,
)


def bitmask_spmv(
    M_values: torch.Tensor,
    M_mask: torch.Tensor,
    M_row_indices: torch.Tensor,
    M_rows: int,
    M_cols: int,
    V: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 0.0,
) -> torch.Tensor:
    """
    Sparse matrix‑vector multiplication: C = alpha * (bitmask_matrix @ V) + beta * C

    Args:
        M_values: non‑zero values, float16, shape (nnz,)
        M_mask:  bitmask, uint8, shape (M_rows, ceil(M_cols/8))
        M_row_indices: int32, shape (M_rows+1,)  (CSR row pointer)
        M_rows:  number of rows
        M_cols:  number of columns
        V:       dense vector, float16, shape (M_cols,)
        alpha, beta: scalar floats (default: alpha=1, beta=0)
    Returns:
        C: float16 tensor of shape (M_rows,)
    """
    M_values = M_values.contiguous().cuda()
    M_mask = M_mask.contiguous().cuda()
    M_row_indices = M_row_indices.contiguous().cuda()
    V = V.contiguous().cuda()

    return _bitmask_spmv_module.bitmask_spmv_launch(
        M_values, M_mask, M_row_indices, M_rows, M_cols, V, alpha, beta
    )
#Macko SPMV:

LIB_NAME = "sparse_spmv"

pkg_version = version(LIB_NAME).replace(".", "_")
cuda_resources = resources.files(f"sparse_spmv.cuda")

BUILD_DIRECTORY = os.environ.get("MACKO_SPMV_BUILD_DIRECTORY", "")

def __init_compressor():
    compressor_source_code = cuda_resources.joinpath("convert_macko.cuh").read_text()
    launcher_source_code = cuda_resources.joinpath(
        "convert_macko_torch_launcher.cuh"
    ).read_text()

    lib_name = f"macko_spmv_compression_{pkg_version}"
    build_directory = None if not BUILD_DIRECTORY else os.path.join(BUILD_DIRECTORY, lib_name)
    if build_directory is not None:
        os.makedirs(build_directory, exist_ok=True)

    lib = load_inline(
        name=lib_name,
        cpp_sources=[compressor_source_code + "\n" + launcher_source_code],
        functions=["cpu_compress"],
        verbose=False,
        with_cuda=True,
        build_directory=build_directory,
    )
    return lib


__compressor_lib = __init_compressor()


def __init_multiply():
    kernel_source_code = cuda_resources.joinpath("macko_spmv.cuh").read_text()
    launcher_source_code = cuda_resources.joinpath(
        "macko_spmv_torch_launchers.cuh"
    ).read_text()

    full_source_code = kernel_source_code + "\n" + launcher_source_code

    cpp_source = """
at::Tensor macko_spmv_launcher(
    at::Tensor M_values,
    at::Tensor M_deltas,
    at::Tensor M_row_indices,
    int64_t M_rows, int64_t M_cols,
    at::Tensor V
);

TORCH_LIBRARY(macko_spmv, m) {
    m.def("multiply(Tensor M_values, Tensor M_deltas, Tensor M_row_indices, \
            int M_rows, int M_cols, Tensor V) -> Tensor");
}

TORCH_LIBRARY_IMPL(macko_spmv, CUDA, m) {
    m.impl("multiply", &macko_spmv_launcher);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("macko_spmv_launcher", \
            torch::wrap_pybind_function(macko_spmv_launcher), \
            "macko_spmv_launcher");
}
    """

    lib_name = f"macko_spmv_multiplication_{pkg_version}"
    build_directory = None if not BUILD_DIRECTORY else os.path.join(BUILD_DIRECTORY, lib_name)
    if build_directory is not None:
        os.makedirs(build_directory, exist_ok=True)

    lib = load_inline(
        name=lib_name,
        cpp_sources=cpp_source,
        cuda_sources=full_source_code,
        functions=None,
        verbose=False,
        with_cuda=True,
        extra_cuda_cflags=["-O3"],
        build_directory=build_directory,
    )
    return lib


__multiply_lib = __init_multiply()


def __torch_registration():
    build_directory = __multiply_lib.__file__

    assert (
        os.path.isfile(build_directory) == 1
    ), f"Expected one _C*.so file, found {build_directory}"
    torch.ops.load_library(__multiply_lib.__file__)


__torch_registration()


@torch.library.register_fake("macko_spmv::multiply")
def _(a, b, c, d, e, f):
    return torch.empty((d,), device=a.device, dtype=a.dtype)


def move_to_device(compressed, device):
    return (
        compressed[0].to(device=device),
        compressed[1].to(device=device),
        compressed[2].to(device=device),
        compressed[3],
        compressed[4],
    )


def compress(M):
    assert M.is_contiguous()
    if M.device.type == "cuda":
        # TODO: implement properly fast gpu only compression
        compressed = __compressor_lib.cpu_compress(M.to("cpu"))
        return move_to_device(compressed, "cuda")

    elif M.device.type == "cpu":
        compressed = __compressor_lib.cpu_compress(M)
        return compressed
    else:
        raise NotImplementedError()


def multiply(compressed_M, V):
    assert compressed_M[0].is_cuda
    assert compressed_M[1].is_cuda
    assert compressed_M[2].is_cuda
    assert V.is_cuda
    assert V.is_contiguous()

    # __multiply_lib.macko_spmv_launcher is also usable
    return torch.ops.macko_spmv.multiply.default(
        compressed_M[0],
        compressed_M[1],
        compressed_M[2],
        compressed_M[3],
        compressed_M[4],
        V,
    )
