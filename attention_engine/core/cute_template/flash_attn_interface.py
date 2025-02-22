# Copyright (c) 2023, Tri Dao.

from typing import Optional, Union

import torch
import torch.nn as nn

# isort: off
# We need to import the CUDA kernels after importing torch

# import flashattn_hopper_cuda
import os
import sys
import subprocess
from pathlib import Path
from packaging.version import parse, Version
import warnings

import torch.utils.cpp_extension
from torch.utils.cpp_extension import CUDA_HOME


include_path = [

]

# ninja build does not work unless include_dirs are abs path
this_dir = os.path.dirname(os.path.abspath(__file__))

PACKAGE_NAME = "flashattn-hopper"

BASE_WHEEL_URL = "https://github.com/Dao-AILab/flash-attention/releases/download/{tag_name}/{wheel_name}"

# FORCE_BUILD: Force a fresh build locally, instead of attempting to find prebuilt wheels
# SKIP_CUDA_BUILD: Intended to allow CI to use a simple `python setup.py sdist` run to copy over raw files, without any cuda compilation
FORCE_BUILD = os.getenv("FAHOPPER_FORCE_BUILD", "FALSE") == "TRUE"
SKIP_CUDA_BUILD = os.getenv("FAHOPPER_SKIP_CUDA_BUILD", "FALSE") == "TRUE"
# For CI, we want the option to build with C++11 ABI since the nvcr images use C++11 ABI
FORCE_CXX11_ABI = os.getenv("FAHOPPER_FORCE_CXX11_ABI", "FALSE") == "TRUE"

# export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so
os.environ["LD_PRELOAD"] = "/usr/lib/x86_64-linux-gnu/libcuda.so"

def get_platform():
    """
    Returns the platform name as used in wheel filenames.
    """
    if sys.platform.startswith("linux"):
        return "linux_x86_64"
    elif sys.platform == "darwin":
        mac_version = ".".join(platform.mac_ver()[0].split(".")[:2])
        return f"macosx_{mac_version}_x86_64"
    elif sys.platform == "win32":
        return "win_amd64"
    else:
        raise ValueError("Unsupported platform: {}".format(sys.platform))


def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_version = parse(output[release_idx].split(",")[0])

    return raw_output, bare_metal_version


def check_if_cuda_home_none(global_option: str) -> None:
    if CUDA_HOME is not None:
        return
    # warn instead of error because user could be downloading prebuilt wheels, so nvcc won't be necessary
    # in that case.
    warnings.warn(
        f"{global_option} was requested, but nvcc was not found.  Are you sure your environment has nvcc available?  "
        "If you're installing within a container from https://hub.docker.com/r/pytorch/pytorch, "
        "only images whose names contain 'devel' will provide nvcc."
    )


def append_nvcc_threads(nvcc_extra_args):
    return nvcc_extra_args + ["--threads", "4"]

if not SKIP_CUDA_BUILD:
    print("\n\ntorch.__version__  = {}\n\n".format(torch.__version__))
    TORCH_MAJOR = int(torch.__version__.split(".")[0])
    TORCH_MINOR = int(torch.__version__.split(".")[1])

    check_if_cuda_home_none("--fahopper")
    cc_flag = []
    _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
    if bare_metal_version < Version("12.3"):
        raise RuntimeError("FA Hopper is only supported on CUDA 12.3 and above")
    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_90a,code=sm_90a")

    # HACK: The compiler flag -D_GLIBCXX_USE_CXX11_ABI is set to be the same as
    # torch._C._GLIBCXX_USE_CXX11_ABI
    # https://github.com/pytorch/pytorch/blob/8472c24e3b5b60150096486616d98b7bea01500b/torch/utils/cpp_extension.py#L920
    if FORCE_CXX11_ABI:
        torch._C._GLIBCXX_USE_CXX11_ABI = True
    repo_dir = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../"))
    cutlass_dir = repo_dir / "3rd_parties" / "cutlass"
    sources = [
        os.path.join(os.path.dirname(__file__), "flash_api.cpp"),
        os.path.join(os.path.dirname(__file__), "flash_fwd.cu"),
        os.path.join(os.path.dirname(__file__), "flash_bwd.cu"),
    ]
    nvcc_flags = [
        "-O3",
        # "-O0",
        "-std=c++17",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        "--ptxas-options=-v",  # printing out number of registers
        "--ptxas-options=--verbose,--register-usage-level=10,--warn-on-local-memory-usage",  # printing out number of registers
        "-lineinfo",
        "-DCUTLASS_DEBUG_TRACE_LEVEL=0",  # Can toggle for debugging
        "-DNDEBUG",  # Important, otherwise performance is severely impacted             
    ]
    include_dirs = [
        # Path(this_dir) / "fmha-pipeline",
        # repo_dir / "lib",
        # repo_dir / "include",
        str(cutlass_dir / "include"),
        # cutlass_dir / "examples" / "common",
        # cutlass_dir / "tools" / "util" / "include",
    ]

    flashattn_hopper_cuda = torch.utils.cpp_extension.load(
        name="flashattn_hopper_cuda"+"{{dimqk}}_{{dimv}}_{{cutlass_dtype}}".replace("::", "_").replace(" ", "_"),
        sources=sources,
        extra_cflags=[
            "-O3", "-std=c++17",
        ],
        extra_cuda_cflags=append_nvcc_threads(
                    nvcc_flags + cc_flag
                ),
        extra_include_paths=include_dirs,
        with_cuda=True,
    )


# isort: on

def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x

def _flash_attn_forward(q, k, v,{{custom_tensors}} softmax_scale, causal, descale_q = None, descale_k = None, descale_v = None):
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    # TODO: final_rowscales softmax_lse, S_dmask
    # out, q, k, v, out_padded, {{final_rowscales_list}} S_dmask = flashattn_hopper_cuda.fwd(
    out_res = flashattn_hopper_cuda.fwd(
        q,
        k,
        v,
        None,
        softmax_scale,
        descale_q,
        descale_k,
        descale_v,
        {{custom_tensors}}
        causal,
    )
    # return out, q, k, v, out_padded, softmax_lse, S_dmask
    return out_res


def _flash_attn_backward(
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    dq,
    dk,
    dv,
    softmax_scale,
    causal,
    deterministic=False
):
    # dq, dk, dv are allocated by us so they should already be contiguous
    dout, q, k, v, out = [maybe_contiguous(x) for x in (dout, q, k, v, out)]
    dq, dk, dv, softmax_d, *rest = flashattn_hopper_cuda.bwd(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        dq,
        dk,
        dv,
        softmax_scale,
        causal,
        deterministic,
    )
    return dq, dk, dv, softmax_d


class FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        {{custom_tensors}}
        softmax_scale,
        causal,
        deterministic=False,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        # out, q, k, v, out_padded, softmax_lse, S_dmask = _flash_attn_forward(
        out_res = _flash_attn_forward(
            q,
            k,
            v,
            {{custom_tensors}}
            softmax_scale,
            causal
        )
        # ctx.save_for_backward(q, k, v, out_padded, softmax_lse)
        ctx.save_for_backward(*(out_res[1:-1]))
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.deterministic = deterministic
        # return out#, softmax_lse
        return out_res[0]#, *out_res[5:-1]

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse = ctx.saved_tensors
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        _flash_attn_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            dq,
            dk,
            dv,
            ctx.softmax_scale,
            ctx.causal,
            ctx.deterministic,
        )
        dq = dq[..., : dout.shape[-1]]  # We could have padded the head dimension
        dk = dk[..., : dout.shape[-1]]
        dv = dv[..., : dout.shape[-1]]
        return dq, dk, dv, None, None, None


def flash_attn_func(
    q,
    k,
    v,
    {{custom_tensors}}
    softmax_scale=None,
    causal=False,
    deterministic=False
):
    """dropout_p should be set to 0.0 during evaluation
    Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If the row of the mask is all zero, the output will be zero.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

    Arguments:
        q: (batch_size, seqlen, nheads, headdim)
        k: (batch_size, seqlen, nheads_k, headdim)
        v: (batch_size, seqlen, nheads_k, headdim)
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (batch_size, seqlen, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
        S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen).
            The output of softmax (possibly with different scaling). It also encodes the dropout
            pattern (negative means that location was dropped, nonnegative means it was kept).
    """
    return FlashAttnFunc.apply(
        q,
        k,
        v,
        {{custom_tensors}}
        softmax_scale,
        causal,
        deterministic,
    )
