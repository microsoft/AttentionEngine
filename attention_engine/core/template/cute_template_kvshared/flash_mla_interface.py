from typing import Optional, Tuple

import torch

# import flash_mla_cuda
import torch.utils.cpp_extension
import os
from pathlib import Path

cc_flag = []
cc_flag.append("-gencode")
cc_flag.append("arch=compute_90a,code=sm_90a")

repo_dir = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../"))
cutlass_dir = repo_dir / "3rd_parties" / "cutlass_39"

sources = [
    os.path.join(os.path.dirname(__file__), "flash_api.cpp"),
    os.path.join(os.path.dirname(__file__), "kernels", "get_mla_metadata.cu"),
    os.path.join(os.path.dirname(__file__), "kernels", "mla_combine.cu"),
    os.path.join(os.path.dirname(__file__), "kernels", "splitkv_mla.cu"),
]
flash_mla_cuda = torch.utils.cpp_extension.load(
    name="flash_mla_hopper_cuda"+"{{dimqk}}_{{dimv}}_{{cutlass_dtype}}".replace("::", "_").replace(" ", "_"),
    sources=sources,
    extra_cflags=[
        "-O3", "-std=c++17", "-DNDEBUG", "-Wno-deprecated-declarations",
        # "-DFLASH_MLA_DISABLE_FP16"
    ],
    extra_cuda_cflags=[
        "-O3",
        "-std=c++17",
        "-DNDEBUG",
        "-D_USE_MATH_DEFINES",
        "-Wno-deprecated-declarations",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        "--ptxas-options=-v,--register-usage-level=10",
        # "-DFLASH_MLA_DISABLE_FP16",
        "-lineinfo"
    ] + cc_flag,
    extra_include_paths=[
        str(cutlass_dir / "include"),
    ],
    with_cuda=True,
    # build_directory=os.path.expanduser("~/.cache/torch_extensions/flash_mla_cuda"),
)


def get_mla_metadata(
    cache_seqlens: torch.Tensor,
    num_heads_per_head_k: int,
    num_heads_k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Arguments:
        cache_seqlens: (batch_size), dtype torch.int32.
        num_heads_per_head_k: Equals to seq_len_q * num_heads_q // num_heads_k.
        num_heads_k: num_heads_k.

    Returns:
        tile_scheduler_metadata: (num_sm_parts, TileSchedulerMetaDataSize), dtype torch.int32.
        num_splits: (batch_size + 1), dtype torch.int32.
    """
    return flash_mla_cuda.get_mla_metadata(cache_seqlens, num_heads_per_head_k, num_heads_k)


def flash_mla_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    head_dim_v: int,
    tile_scheduler_metadata: torch.Tensor,
    num_splits: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Arguments:
        q: (batch_size, seq_len_q, num_heads_q, head_dim).
        k_cache: (num_blocks, page_block_size, num_heads_k, head_dim).
        block_table: (batch_size, max_num_blocks_per_seq), torch.int32.
        cache_seqlens: (batch_size), torch.int32.
        head_dim_v: Head dimension of v.
        tile_scheduler_metadata: (num_sm_parts, TileSchedulerMetaDataSize), torch.int32, returned by get_mla_metadata.
        num_splits: (batch_size + 1), torch.int32, returned by get_mla_metadata.
        softmax_scale: float. The scale of QK^T before applying softmax. Default to 1 / sqrt(head_dim).
        causal: bool. Whether to apply causal attention mask.

    Returns:
        out: (batch_size, seq_len_q, num_heads_q, head_dim_v).
        softmax_lse: (batch_size, num_heads_q, seq_len_q), torch.float32.
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    out, softmax_lse = flash_mla_cuda.fwd_kvcache_mla(
        q,
        k_cache,
        head_dim_v,
        cache_seqlens,
        block_table,
        softmax_scale,
        causal,
        tile_scheduler_metadata,
        num_splits,
    )
    return out, softmax_lse
