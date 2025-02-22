import torch
import math
import torch.nn.functional as F

import torch
import matplotlib.pyplot as plt

from functools import lru_cache, partial
import functools


def do_bench(
    fn,
    warmup=25,
    rep=100,
    _n_warmup=0,
    _n_repeat=0,
    grad_to_none=None,
    quantiles=None,
    fast_flush=True,
    return_mode="mean",
):
    """
    Benchmark the runtime of the provided function. By default, return the median runtime of :code:`fn` along with
    the 20-th and 80-th performance percentile.

    :param fn: Function to benchmark
    :type fn: Callable
    :param warmup: Warmup time (in ms)
    :type warmup: int
    :param rep: Repetition time (in ms)
    :type rep: int
    :param grad_to_none: Reset the gradient of the provided tensor to None
    :type grad_to_none: torch.tensor, optional
    :param quantiles: Performance percentile to return in addition to the median.
    :type quantiles: list[float]
    :param fast_flush: Use faster kernel to flush L2 between measurements
    :type fast_flush: bool
    """
    assert return_mode in ["min", "max", "mean", "median"]
    fn()
    torch.cuda.synchronize()

    # We maintain a buffer of 256 MB that we clear
    # before each kernel call to make sure that the L2
    # doesn't contain any input data before the run
    if fast_flush:
        cache = torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda")
    else:
        cache = torch.empty(int(256e6), dtype=torch.int8, device="cuda")

    # Estimate the runtime of the function
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(5):
        cache.zero_()
        fn()
    end_event.record()
    torch.cuda.synchronize()
    estimate_ms = start_event.elapsed_time(end_event) / 5

    # compute number of warmup and repeat
    n_warmup = max(1, int(warmup / estimate_ms))
    n_repeat = max(1, int(rep / estimate_ms))
    if _n_warmup > 0:
        n_warmup = _n_warmup
    if _n_repeat > 0:
        n_repeat = _n_repeat
    # start_event = [torch.cuda.Event(enable_timing=True) for i in range(n_repeat)]
    # end_event = [torch.cuda.Event(enable_timing=True) for i in range(n_repeat)]
    # Warm-up
    for _ in range(n_warmup):
        fn()
    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for i in range(n_repeat):
        # we don't want `fn` to accumulate gradient values
        # if it contains a backward pass. So we clear the
        # provided gradients
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None
        # we clear the L2 cache before each run
        # cache.zero_()
        # record time of `fn`
        # start_event[i].record()
        fn()
        # end_event[i].record()
    end.record()
    # Record clocks
    torch.cuda.synchronize()
    time = start.elapsed_time(end)
    time = time / n_repeat
    return time


def analysis_tensor_data(a: torch.Tensor, plot: bool = False,
                         figure_name: str = 'tensor_distribution.png'):
    print(f"Data type: {a.dtype}")
    print(f"Device: {a.device}")
    print(f"Shape: {a.shape}")

    max_value = a.max()
    min_value = a.min()
    mean_value = a.mean()
    std_value = a.std()
    median_value = a.median()
    # quantiles = torch.quantile(a.float(), torch.tensor([0.25, 0.5, 0.75], device=a.device))
    histogram = torch.histc(a.float(), bins=50, min=0, max=1)

    if plot:
        plt.hist(a.float().cpu().detach().numpy().flatten(), bins=50)
        plt.title('Tensor Distribution')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.show()
        plt.savefig(figure_name)

    print(f"Max: {max_value}")
    print(f"Min: {min_value}")
    print(f"Mean: {mean_value}")
    print(f"Std: {std_value}")
    print(f"Median: {median_value}")
    # print(f"Quantiles 4: {quantiles}")
    print(f"Histogram: {histogram}")


def check_close(o, O_ref, rtol=1e-3, atol=1e-3):
    absolute_error = torch.abs(o - O_ref)
    relative_error = absolute_error / (torch.abs(O_ref) + 1e-6)

    tolerance_check = (absolute_error < atol) | (relative_error < rtol)

    num_not_close = (~tolerance_check).sum().item()
    total_elements = o.numel()
    percentage_not_close = (num_not_close / total_elements) * 100
    print(f"{percentage_not_close:.2f}% of the elements are not close.")
    result = torch.all(tolerance_check)
    if not result:
        num_not_close = (~tolerance_check).sum().item()
        total_elements = o.numel()
        percentage_not_close = (num_not_close / total_elements) * 100
        print(f"{percentage_not_close:.2f}% of the elements are not close.")
        print("Error elements:")
        error_indices = torch.nonzero(~tolerance_check, as_tuple=False)
        for idx in error_indices[:10]:
            print(
                f"Index: {tuple(idx.cpu().numpy())}, Reference: {O_ref[tuple(idx.cpu().numpy())]}, Library: {o[tuple(idx.cpu().numpy())]}")

    return result


def print_debug(o, O_ref, rtol=1e-3, atol=1e-3, save_file=True):
    close_mask = torch.isclose(o, O_ref, rtol=rtol, atol=atol)
    total_elements = o.numel()
    num_not_close = (~close_mask).sum().item()
    percentage_not_close = (num_not_close / total_elements) * 100
    print(f"{percentage_not_close:.2f}% of the elements are not close.")
    print(
        f"Total elements: {total_elements}, Not close elements: {num_not_close}")
    # max diff and idx
    max_diff = (o - O_ref).abs().max().item()
    max_diff_idx = (o - O_ref).abs().argmax().item()
    max_diff_idx = torch.unravel_index(torch.tensor(max_diff_idx), o.shape)
    print(f"Max diff: {max_diff} at index {max_diff_idx}")
    print(f"Reference: {O_ref[max_diff_idx]}")
    print(f"Library: {o[max_diff_idx]}")
    print(torch.allclose(o, O_ref, rtol=rtol, atol=atol))
    # max relative diff and idx
    max_rel_diff = ((o - O_ref).abs() / O_ref.abs()).max().item()
    max_rel_diff_idx = ((o - O_ref).abs() / O_ref.abs()).argmax().item()
    max_rel_diff_idx = torch.unravel_index(
        torch.tensor(max_rel_diff_idx), o.shape)
    print(f"Max rel diff: {max_rel_diff} at index {max_rel_diff_idx}")
    print(f"Reference: {O_ref[max_rel_diff_idx]}")
    print(f"Library: {o[max_rel_diff_idx]}")

    if save_file:
        with open("o_ref.txt", "w") as f:
            O_ref_1 = O_ref.cpu()
            for idx, element in enumerate(O_ref_1):  # .flatten()):
                f.write(f"{idx}: {element}\n")
        with open("o.txt", "w") as f:
            o_1 = o.cpu()
            for idx, element in enumerate(o_1):  # .flatten()):
                f.write(f"{idx}: {element}\n")


# def bench_func_fwd(attn, B, H, S, D, DV, custom_fwd_input={}, causal=True, dtype=torch.float16):
#     tflops = 2 * B * H * S * S * D + 2 * B * H * S * S * DV
#     if causal:
#         tflops = tflops * 0.5
#     bwd_tflops = 4 * B * H * S * S * DV + 6 * B * H * S * S * D
#     if causal:
#         bwd_tflops = bwd_tflops * 0.5
#     torch.cuda.manual_seed(0)
#     dtype = dtype
#     device = "cuda"
#     accum_dtype = torch.float32
#     query = torch.randn(
#         B, S, H, D, device=device, dtype=dtype, requires_grad=True
#     )
#     key = torch.randn(
#         B, S, H, D, device=device, dtype=dtype, requires_grad=True
#     )
#     value = torch.randn(
#         B, S, H, DV, device=device, dtype=dtype, requires_grad=True
#     )
#     do = torch.randn(
#         B, S, H, DV, device=device, dtype=dtype, requires_grad=True
#     )
#     # softmax_bias = 0.1*torch.randn(1, device=device, dtype=torch.float, requires_grad=False)
#     custom_tensors = []
#     for key, value in custom_fwd_input.items():
#         # ["1", "heads", "seq_len", "seq_len_kv"]
#         # -> (1, H, S, S)
#         TENSOR_SHAPE_MAP = [
#             B, H, S, S
#         ]
#         tensor_shape = [

#         ]

#     o = attn(query, key, value, softmax_bias)


#     from flash_sigmoid import flash_attn_func

#     o_ref = flash_attn_func(query, key, value, softmax_scale=1.0,causal=True, sigmoid_bias=softmax_bias_cpu)
#     # print_debug(o, o_ref)

#     from tilelang.profiler import do_bench
#     def run():
#         o = attn(query, key, value, softmax_bias)

#     def run_ref():
#         o_ref = flash_attn_func(query, key, value, softmax_scale=1.0,causal=True, sigmoid_bias=softmax_bias_cpu)

#     do_bench(run)
#     do_bench(run_ref)

#     latency = do_bench(run, warmup=500,rep=1000)
#     print("tl: {:.2f} ms".format(latency))
#     print("tflops: {:.2f}".format(tflops/latency*1e-9))

#     latency_ref = do_bench(run_ref, warmup=500,rep=1000)
#     print("flash: {:.2f} ms".format(latency_ref))
#     print("tflops: {:.2f}".format(tflops/latency_ref*1e-9))
# return latency, (tflops/latency*1e-9), latency_ref,
# (tflops/latency_ref*1e-9)


def do_bench_mamba(linear_attention, B, HQ, HK, H, TLen,
                   D, DV, BT, requires_grad=False):
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined, ssd_chunk_scan_combined_ref
    torch.cuda.manual_seed(0)
    # torch.cuda.set_device(1)
    # B, H, D, DV = 16, 8, 128, 128
    # TLen = 16384 # 512
    # BT= 64
    dtype = torch.bfloat16
    accum_dtype = torch.float32
    device = "cuda"

    X_mamba = 0.4 * torch.randn(B, TLen, H, DV, dtype=dtype, device=device)
    dt_mamba = 0.7 * torch.randn(B, TLen, H, dtype=accum_dtype, device=device)
    A_mamba = 1.5 * torch.randn(H, dtype=dtype, device=device) - 4
    B_mamba = 0.8 * torch.randn(B, TLen, HK, D, dtype=dtype, device=device)
    C_mamba = torch.randn(B, TLen, HQ, D, dtype=dtype, device=device)
    if requires_grad:
        do_mamba = 0.1 * torch.randn(B, TLen, H,
                                     DV, dtype=dtype, device=device)

    # initialize dt_bias
    factory_kwargs = {"device": device, "dtype": dtype}
    dt_min = 0.001
    dt_max = 0.1
    dt = torch.exp(
        torch.rand(H, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
        + math.log(dt_min)
    )
    dt = torch.clamp(dt, min=1e-4)
    dt_bias_mamba = dt + torch.log(-torch.expm1(-dt))

    # compute dt_mamba(not fused !!)
    dt_mamba = F.softplus(dt_mamba + dt_bias_mamba)

    # print(dt_bias_mamba.max().item(), dt_bias_mamba.min().item())
    # print(dt_mamba.max().item(), dt_mamba.min().item())
    # print(A_mamba.max().item(), A_mamba.min().item())

    X_mamba.detach_().requires_grad_(requires_grad)
    dt_mamba.detach_().requires_grad_(requires_grad)
    A_mamba.detach_().requires_grad_(requires_grad)
    B_mamba.detach_().requires_grad_(requires_grad)
    C_mamba.detach_().requires_grad_(requires_grad)

    out_ref = mamba_chunk_scan_combined(
        X_mamba, dt_mamba, A_mamba, B_mamba, C_mamba,
        chunk_size=BT, D=None, return_final_states=False,
        dt_bias=None  # dt_bias_mamba
    )
    # ssd_chunk_scan_combined_ref = torch.compile(ssd_chunk_scan_combined_ref)
    # out_ref = ssd_chunk_scan_combined_ref(
    #     X_mamba, dt_mamba, A_mamba, B_mamba, C_mamba,
    #     chunk_size=BT, D=None,
    #     dt_bias=None# dt_bias_mamba
    # )
    # print(out_ref)

    q = C_mamba.clone().transpose(1, 2).contiguous()
    k = B_mamba.clone().transpose(1, 2).contiguous()
    v = X_mamba.clone().transpose(1, 2).contiguous()
    A_mamba1 = A_mamba[None, :].clone().contiguous()
    dt_mamba1 = dt_mamba.clone().transpose(1, 2).contiguous()
    if requires_grad:
        do_mamba1 = do_mamba.clone().transpose(1, 2).contiguous()

    q = q.detach().requires_grad_(requires_grad)
    k = k.detach().requires_grad_(requires_grad)
    v = v.detach().requires_grad_(requires_grad)
    A_mamba1 = A_mamba1.detach().requires_grad_(requires_grad)
    dt_mamba1 = dt_mamba1.detach().requires_grad_(requires_grad)

    out = linear_attention(
        q, k, v, dt_mamba1, A_mamba1, dt_mamba1.bfloat16()
    )
    if requires_grad:
        out.backward(do_mamba1, retain_graph=True)
        out_ref.backward(do_mamba, retain_graph=True)

    out2 = out.transpose(1, 2)
    print_debug(out2, out_ref, rtol=1e-2, atol=1e-2)
    # torch.testing.assert_close(out, out_ref, rtol=1e-2, atol=4e-2)
    # assert check_close(out, out_ref, rtol=5e-2, atol=4e-2)

    if requires_grad:
        print_debug(q.grad.transpose(1, 2), C_mamba.grad, rtol=1e-2, atol=1e-2)
        print_debug(k.grad.transpose(1, 2), B_mamba.grad, rtol=1e-2, atol=1e-2)
        print_debug(v.grad.transpose(1, 2), X_mamba.grad, rtol=1e-2, atol=1e-2)
        print_debug(A_mamba1.grad, A_mamba.grad[None, :], rtol=1e-2, atol=1e-2)
        print_debug(
            dt_mamba1.grad.transpose(
                1,
                2),
            dt_mamba.grad,
            rtol=1e-2,
            atol=1e-2)

    from tilelang.profiler import do_bench

    def run():
        out = linear_attention(
            q, k, v, dt_mamba1, A_mamba1, dt_mamba1.bfloat16()
        )

    def run_ref():
        out_ref = mamba_chunk_scan_combined(
            X_mamba, dt_mamba, A_mamba, B_mamba, C_mamba,
            chunk_size=BT, D=None, return_final_states=False,
            dt_bias=None  # dt_bias_mamba
        )
    #     out_ref = ssd_chunk_scan_combined_ref(
    #         X_mamba, dt_mamba, A_mamba, B_mamba, C_mamba,
    #         chunk_size=BT, D=None, # return_final_states=False,
    #         dt_bias=None# dt_bias_mamba
    #     )

    def run_backward():
        out.backward(do_mamba1, retain_graph=True)

    def run_ref_backward():
        out_ref.backward(do_mamba, retain_graph=True)

    do_bench(run)
    do_bench(run_ref)

    latency = do_bench(run, warmup=100, rep=100)
    print("tl: {:.5f} ms".format(latency))

    latency = do_bench(run_ref, warmup=100, rep=100)
    print("MAMBA2: {:.5f} ms".format(latency))

    if requires_grad:

        latency = do_bench(run_backward, warmup=100, rep=100)
        print("tl: {:.5f} ms".format(latency))

        latency = do_bench(run_ref_backward, warmup=100, rep=100)
        print("MAMBA2: {:.5f} ms".format(latency))


def test_mamba_simple_gla():
    B, H, TLen, D, DV, BT = 16, 8, 512, 128, 128, 64
    from fla.ops.simple_gla import chunk_simple_gla
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

    torch.cuda.manual_seed(0)

    dtype = torch.bfloat16
    accum_dtype = torch.float32
    device = "cuda"

    X_mamba = 0.4 * torch.randn(B, TLen, H, D, dtype=dtype, device=device)
    X_mamba = X_mamba.detach().requires_grad_()
    dt_mamba = 0.7 * torch.randn(B, TLen, H, dtype=accum_dtype, device=device)
    dt_mamba = dt_mamba.detach().requires_grad_()
    A_mamba = 1.5 * torch.randn(H, dtype=dtype, device=device) - 4
    A_mamba = A_mamba.detach().requires_grad_()
    B_mamba = 0.8 * torch.randn(B, TLen, H, D, dtype=dtype, device=device)
    B_mamba = B_mamba.detach().requires_grad_()
    C_mamba = torch.randn(
        B,
        TLen,
        H,
        D,
        dtype=dtype,
        device=device,
        requires_grad=True)
    do_mamba = 0.1 * torch.randn(B, TLen, H, DV, dtype=dtype, device=device)

    factory_kwargs = {"device": device, "dtype": dtype}
    dt_min = 0.001
    dt_max = 0.1
    dt = torch.exp(
        torch.rand(H, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
        + math.log(dt_min)
    )
    dt = torch.clamp(dt, min=1e-4)
    dt_bias_mamba = dt + torch.log(-torch.expm1(-dt))

    # compute dt_mamba(not fused !!)
    dt_mamba = F.softplus(dt_mamba + dt_bias_mamba)
    # tmp: test!!! nan
    # dt_mamba = 0.7*torch.randn(B, TLen, H, dtype=accum_dtype, device=device)
    dt_mamba = dt_mamba.detach().requires_grad_()

    out_ref = mamba_chunk_scan_combined(
        X_mamba, dt_mamba, A_mamba, B_mamba, C_mamba,
        chunk_size=BT, D=None, return_final_states=False,
        dt_bias=None  # dt_bias_mamba
    )
    out_ref.backward(do_mamba)

    # print(out_ref)

    q = C_mamba.clone().detach().transpose(1, 2).contiguous().requires_grad_()
    k = B_mamba.clone().detach().transpose(1, 2).contiguous().requires_grad_()
    v = X_mamba.clone().detach().transpose(1, 2).contiguous().requires_grad_()
    # g = (A_mamba * dt_mamba).transpose(1, 2).contiguous()
    A_mamba1 = A_mamba[None, :].clone().detach().contiguous().requires_grad_()
    dt_bias_mamba1 = dt_bias_mamba[:, None].contiguous()
    dt_mamba1 = dt_mamba.clone().detach().transpose(
        1, 2).contiguous().requires_grad_()
    dt_mamba1_k = dt_mamba1.clone().detach().bfloat16().requires_grad_()

    do_mamba1 = do_mamba.transpose(1, 2).contiguous()
    out, _ = chunk_simple_gla(
        # gla dt_mamba1 is bf16
        q, k * dt_mamba1_k[..., None], v, dt_mamba1_k * A_mamba1[..., None],
        scale=1.0, output_final_state=False
    )
    out.backward(do_mamba1)
    out = out.transpose(1, 2).contiguous()

    print_debug(out, out_ref)
    # print("out_ref", out_ref)
    # print("out", out)
    # analysis_tensor_data(out_ref, plot=True, figure_name='tensor_distribution_out_ref.png')
    # analysis_tensor_data(out, plot=True, figure_name='tensor_distribution_out.png')

    # torch.testing.assert_close(out, out_ref, rtol=1e-2, atol=4e-2)
    assert check_close(out, out_ref, rtol=5e-2, atol=4e-2)

    # print("q.grad", q.grad)
    # print("C_mamba.grad", C_mamba.grad)
    q_grad = q.grad.transpose(1, 2)
    k_grad = k.grad.transpose(1, 2)
    v_grad = v.grad.transpose(1, 2)
    dt_mamba1_grad = dt_mamba1_k.grad.transpose(1, 2)
    A_mamba1_grad = A_mamba1.grad.squeeze(0)

    assert check_close(q_grad, C_mamba.grad, rtol=1e-2, atol=4e-2)
    assert check_close(k_grad, B_mamba.grad, rtol=5e-2, atol=4e-2)
    assert check_close(v_grad, X_mamba.grad, rtol=5e-2, atol=4e-2)
    # here ???
    assert check_close(dt_mamba1_grad, dt_mamba.grad, rtol=5e-2, atol=5e-1)
    # print("dt_mamba1_grad", dt_mamba1_grad)
    # print("dt_mamba.grad", dt_mamba.grad)
    # analysis_tensor_data(dt_mamba1_grad, plot=True, figure_name='tensor_distribution_dt_mamba1_grad01.png')
    # analysis_tensor_data(dt_mamba.grad, plot=True, figure_name='tensor_distribution_dt_mamba_grad01.png')
    assert check_close(A_mamba1_grad, A_mamba.grad, rtol=9e-2, atol=4e-2)

    from tilelang.profiler import do_bench

    def run():
        out, _ = chunk_simple_gla(
            q, k * dt_mamba1_k[..., None], v, dt_mamba1_k *
            A_mamba1[..., None],
            scale=1.0, output_final_state=False
        )
        out.backward(do_mamba1)

    def run_ref():
        out_ref = mamba_chunk_scan_combined(
            X_mamba, dt_mamba, A_mamba, B_mamba, C_mamba,
            chunk_size=BT, D=None, return_final_states=False,
            dt_bias=None  # dt_bias_mamba
        )
        out_ref.backward(do_mamba)

    do_bench(run)
    do_bench(run_ref)

    print("fwd+bwd: ")
    latency = do_bench(run, warmup=500, rep=1000)
    print("triton: {:.5f} ms".format(latency))

    latency = do_bench(run_ref, warmup=500, rep=1000)
    print("MAMBA2: {:.5f} ms".format(latency))


def do_bench_simple_gla(linear_attention, B, H, TLen,
                        D, DV, BT, requires_grad=False):
    torch.cuda.manual_seed(0)
    # torch.cuda.set_device(1)
    # B, H, D, DV = 16, 8, 128, 128
    # TLen = 16384 # 512
    # BT= 64
    dtype = torch.bfloat16
    accum_dtype = torch.float32
    device = "cuda"

    q = torch.randn(B, H, TLen, D, device=device, dtype=dtype)
    k = torch.randn(B, H, TLen, D, device=device, dtype=dtype)
    g = F.logsigmoid(torch.randn(B, H, TLen, device=device,
                     dtype=accum_dtype)).clamp_min(-5)
    v = torch.randn(B, H, TLen, DV, device=device, dtype=dtype)
    do = torch.randn(B, H, TLen, DV, device=device, dtype=dtype)

    q.detach_().requires_grad_(requires_grad)
    k.detach_().requires_grad_(requires_grad)
    g.detach_().requires_grad_(requires_grad)
    v.detach_().requires_grad_(requires_grad)

    from fla.ops.simple_gla import chunk_simple_gla
    from fla.ops.simple_gla.naive import torch_simple_gla
    q1 = q.clone()
    k1 = k.clone()
    v1 = v.clone()
    g1 = g.clone().bfloat16()

    q1.detach_().requires_grad_(requires_grad)
    k1.detach_().requires_grad_(requires_grad)
    g1.detach_().requires_grad_(requires_grad)
    v1.detach_().requires_grad_(requires_grad)

    out_ref, _ = chunk_simple_gla(
        q1, k1, v1, g1, scale=None, output_final_state=False
    )
    # torch_simple_gla = torch.compile(torch_simple_gla)
    # out_ref = torch_simple_gla(
    #         q, k, v, g1, scale=None, chunk_size=512
    # )
    out = linear_attention(
        q, k, v, g
    )
    print_debug(out, out_ref, rtol=1e-2, atol=1e-2)

    if requires_grad:
        out.backward(do, retain_graph=True)
        out_ref.backward(do, retain_graph=True)
        print_debug(q.grad, q1.grad, rtol=1e-2, atol=1e-2)
        print_debug(k.grad, k1.grad, rtol=1e-2, atol=1e-2)
        print_debug(v.grad, v1.grad, rtol=1e-2, atol=1e-2)
        print_debug(g.grad, g1.grad.float(), rtol=1e-2, atol=1e-2)

    from tilelang.profiler import do_bench

    def run():
        out = linear_attention(
            q, k, v, g
        )

    def run_ref():
        out, _ = chunk_simple_gla(
            q, k, v, g1, scale=None, output_final_state=False
        )
        # out = torch_simple_gla(
        #     q, k, v, g1, scale=None, chunk_size=512
        # )

    def run_bacward():
        out.backward(do, retain_graph=True)

    def run_bacward_ref():
        out_ref.backward(do, retain_graph=True)

    latency = do_bench(run, warmup=100, rep=100)
    print("tl: {:.2f} ms".format(latency))
    latency = do_bench(run_ref, warmup=100, rep=100)
    print("simple gla: {:.2f} ms".format(latency))

    if requires_grad:
        latency = do_bench(run_bacward, warmup=100, rep=100)
        print("tl backward: {:.2f} ms".format(latency))
        latency = do_bench(run_bacward_ref, warmup=100, rep=100)
        print("simple gla backward: {:.2f} ms".format(latency))


def do_bench_retention_linear(
        linear_attention, B, H, TLen, D, DV, requires_grad=False):
    torch.cuda.manual_seed(0)
    # torch.cuda.set_device(1)
    # B, H, D, DV = 16, 8, 128, 128
    # TLen = 16384 # 512
    # BT= 64
    dtype = torch.bfloat16
    accum_dtype = torch.float32
    device = "cuda"

    q = torch.randn(B, H, TLen, D, device=device, dtype=dtype)
    k = torch.randn(B, H, TLen, D, device=device, dtype=dtype)
    # g = (1 - torch.exp(torch.linspace(math.log(1/32), math.log(1/512), H, dtype=accum_dtype))).detach()
    g = torch.tensor(range(0, H), dtype=accum_dtype)
    g = 1 - torch.exp2(-5 - g)
    g = g[None, :, None].expand(B, H, TLen).cuda().detach().contiguous()
    v = torch.randn(B, H, TLen, DV, device=device, dtype=dtype)

    q.detach_().requires_grad_(requires_grad)
    k.detach_().requires_grad_(requires_grad)
    g.detach_().requires_grad_(False)
    v.detach_().requires_grad_(requires_grad)

    do = torch.randn(B, H, TLen, DV, device=device, dtype=dtype)
    from fla.ops.retention import chunk_retention
    from fla.ops.retention.naive import naive_retention
    out_ref, _ = chunk_retention(
        q, k, v, scale=None, output_final_state=False
    )
    # naive_retention = torch.compile(naive_retention)
    # out_ref = naive_retention(q,k,v)

    q1 = q.clone()
    k1 = k.clone()
    v1 = v.clone()
    g1 = g.clone()

    q1.detach_().requires_grad_(requires_grad)
    k1.detach_().requires_grad_(requires_grad)
    g1.detach_().requires_grad_(False)
    v1.detach_().requires_grad_(requires_grad)
    out = linear_attention(
        q1, k1, v1, g1
    )
    print_debug(out, out_ref, rtol=1e-2, atol=1e-2)
    if requires_grad:
        out.backward(do, retain_graph=True)
        out_ref.backward(do, retain_graph=True)
        print_debug(q.grad, q1.grad, rtol=1e-2, atol=1e-2)
        print_debug(k.grad, k1.grad, rtol=1e-2, atol=1e-2)
        print_debug(v.grad, v1.grad, rtol=1e-2, atol=1e-2)

    from tilelang.profiler import do_bench

    def run():
        out = linear_attention(
            q, k, v, g
        )

    def run_ref():
        out, _ = chunk_retention(
            q, k, v, scale=None, output_final_state=False
        )
        # out = naive_retention(q,k,v)

    def run_bacward():
        out.backward(do, retain_graph=True)

    def run_bacward_ref():
        out_ref.backward(do, retain_graph=True)

    latency = do_bench(run, warmup=100, rep=100)
    print("tl: {:.2f} ms".format(latency))
    latency = do_bench(run_ref, warmup=100, rep=100)
    print("retention: {:.2f} ms".format(latency))

    if requires_grad:
        latency = do_bench(run_bacward, warmup=100, rep=100)
        print("tl backward: {:.2f} ms".format(latency))
        latency = do_bench(run_bacward_ref, warmup=100, rep=100)
        print("retention backward: {:.2f} ms".format(latency))


def do_bench_sigmoidattn(attn, B, H, S, D, DV,
                         dtype=torch.float16, requires_grad=False):
    tflops = 2 * B * H * S * S * D + 2 * B * H * S * S * DV
    tflops = tflops * 0.5
    bwd_tflops = 4 * B * H * S * S * DV + 6 * B * H * S * S * D
    bwd_tflops = bwd_tflops * 0.5
    torch.cuda.manual_seed(0)
    dtype = dtype
    device = "cuda"
    accum_dtype = torch.float32
    query = torch.randn(
        B, S, H, D, device=device, dtype=dtype, requires_grad=requires_grad
    )
    key = torch.randn(
        B, S, H, D, device=device, dtype=dtype, requires_grad=requires_grad
    )
    value = torch.randn(
        B, S, H, DV, device=device, dtype=dtype, requires_grad=requires_grad
    )
    do = torch.randn(
        B, S, H, DV, device=device, dtype=dtype, requires_grad=False
    )
    softmax_bias = 0.1 * torch.randn(1,
                                     device=device,
                                     dtype=accum_dtype,
                                     requires_grad=False)
    softmax_bias_cpu = softmax_bias.cpu()

    o = attn(query, key, value, softmax_bias)
    if requires_grad:
        o.backward(do, retain_graph=True)
        dQ, query.grad = query.grad.clone(), None
        dK, key.grad = key.grad.clone(), None
        dV, value.grad = value.grad.clone(), None

    from flash_sigmoid import flash_attn_func

    o_ref = flash_attn_func(
        query,
        key,
        value,
        softmax_scale=1.0,
        causal=True,
        sigmoid_bias=softmax_bias_cpu)
    print_debug(o, o_ref)
    if requires_grad:
        o_ref.backward(do, retain_graph=True)
        print_debug(query.grad, dQ)
        print_debug(key.grad, dK)
        print_debug(value.grad, dV)

    from tilelang.profiler import do_bench

    def run():
        o = attn(query, key, value, softmax_bias)

    def run_bacward():
        o.backward(do, retain_graph=True)

    def run_ref():
        o_ref = flash_attn_func(
            query,
            key,
            value,
            softmax_scale=1.0,
            causal=True,
            sigmoid_bias=softmax_bias_cpu)

    def run_ref_backward():
        o_ref.backward(do, retain_graph=True)
    # do_bench(run)
    # do_bench(run_bacward)

    latency = do_bench(run, warmup=500, rep=1000)
    print("tl: {:.2f} ms".format(latency))
    print("tflops: {:.2f}".format(tflops / latency * 1e-9))

    latency = do_bench(run_ref, warmup=500, rep=1000)
    print("flash: {:.2f} ms".format(latency))
    print("tflops: {:.2f}".format(tflops / latency * 1e-9))
    if requires_grad:
        latency = do_bench(run_bacward, warmup=500, rep=1000)
        print("tl bwd: {:.2f} ms".format(latency))
        print("tflops: {:.2f}".format(bwd_tflops / latency * 1e-9))

        latency = do_bench(run_ref_backward, warmup=500, rep=1000)
        print("flash bwd: {:.2f} ms".format(latency))
        print("tflops: {:.2f}".format(bwd_tflops / latency * 1e-9))


def do_bench_sigmoidattn_cute(
        attn, B, H, S, D, DV, dtype=torch.float16, requires_grad=False):
    tflops = 2 * B * H * S * S * D + 2 * B * H * S * S * DV
    tflops = tflops * 0.5
    bwd_tflops = 4 * B * H * S * S * DV + 6 * B * H * S * S * D
    bwd_tflops = bwd_tflops * 0.5
    torch.cuda.manual_seed(0)
    dtype = dtype
    device = "cuda"
    accum_dtype = torch.float32
    query = torch.randn(
        B, S, H, D, device=device, dtype=dtype, requires_grad=requires_grad
    )
    key = torch.randn(
        B, S, H, D, device=device, dtype=dtype, requires_grad=requires_grad
    )
    value = torch.randn(
        B, S, H, DV, device=device, dtype=dtype, requires_grad=requires_grad
    )
    do = torch.randn(
        B, S, H, DV, device=device, dtype=dtype, requires_grad=False
    )
    softmax_bias = 0.1 * torch.randn(1,
                                     device=device,
                                     dtype=accum_dtype,
                                     requires_grad=False).cpu()
    # softmax_bias_cpu = softmax_bias.cpu()

    o = attn(query, key, value, softmax_bias)
    if requires_grad:
        o.backward(do, retain_graph=True)
        dQ, query.grad = query.grad.clone(), None
        dK, key.grad = key.grad.clone(), None
        dV, value.grad = value.grad.clone(), None

    from flash_sigmoid import flash_attn_func

    o_ref = flash_attn_func(
        query,
        key,
        value,
        softmax_scale=1.0,
        causal=True,
        sigmoid_bias=softmax_bias)
    print_debug(o, o_ref)
    if requires_grad:
        o_ref.backward(do, retain_graph=True)
        print_debug(query.grad, dQ)
        print_debug(key.grad, dK)
        print_debug(value.grad, dV)

    from tilelang.profiler import do_bench

    def run():
        o = attn(query, key, value, softmax_bias)

    def run_bacward():
        o.backward(do, retain_graph=True)

    def run_ref():
        o_ref = flash_attn_func(
            query,
            key,
            value,
            softmax_scale=1.0,
            causal=True,
            sigmoid_bias=softmax_bias)

    def run_ref_backward():
        o_ref.backward(do, retain_graph=True)

    latency = do_bench(run, warmup=500, rep=1000)
    print("tl: {:.2f} ms".format(latency))
    print("tflops: {:.2f}".format(tflops / latency * 1e-9))

    latency = do_bench(run_ref, warmup=500, rep=1000)
    print("flash: {:.2f} ms".format(latency))
    print("tflops: {:.2f}".format(tflops / latency * 1e-9))
    if requires_grad:
        latency = do_bench(run_bacward, warmup=500, rep=1000)
        print("tl bwd: {:.2f} ms".format(latency))
        print("tflops: {:.2f}".format(bwd_tflops / latency * 1e-9))

        latency = do_bench(run_ref_backward, warmup=500, rep=1000)
        print("flash bwd: {:.2f} ms".format(latency))
        print("tflops: {:.2f}".format(bwd_tflops / latency * 1e-9))


def bench_sigmoidattn_fwd(attn, B, H, S, D, DV, causal=True):
    tflops = 2 * B * H * S * S * D + 2 * B * H * S * S * DV
    if causal:
        tflops = tflops * 0.5
    bwd_tflops = 4 * B * H * S * S * DV + 6 * B * H * S * S * D
    if causal:
        bwd_tflops = bwd_tflops * 0.5
    torch.cuda.manual_seed(0)
    dtype = torch.float16
    device = "cuda"
    accum_dtype = torch.float32
    query = torch.randn(
        B, S, H, D, device=device, dtype=dtype, requires_grad=True
    )
    key = torch.randn(
        B, S, H, D, device=device, dtype=dtype, requires_grad=True
    )
    value = torch.randn(
        B, S, H, DV, device=device, dtype=dtype, requires_grad=True
    )
    do = torch.randn(
        B, S, H, DV, device=device, dtype=dtype, requires_grad=True
    )
    softmax_bias = 0.1 * torch.randn(1,
                                     device=device,
                                     dtype=torch.float,
                                     requires_grad=False)
    softmax_bias_cpu = softmax_bias.cpu()

    o = attn(query, key, value, softmax_bias)

    from flash_sigmoid import flash_attn_func

    o_ref = flash_attn_func(
        query,
        key,
        value,
        softmax_scale=1.0,
        causal=True,
        sigmoid_bias=softmax_bias_cpu)
    # print_debug(o, o_ref)

    from tilelang.profiler import do_bench

    def run():
        o = attn(query, key, value, softmax_bias)

    def run_ref():
        o_ref = flash_attn_func(
            query,
            key,
            value,
            softmax_scale=1.0,
            causal=True,
            sigmoid_bias=softmax_bias_cpu)

    do_bench(run)
    do_bench(run_ref)

    latency = do_bench(run, warmup=500, rep=1000)
    print("tl: {:.2f} ms".format(latency))
    print("tflops: {:.2f}".format(tflops / latency * 1e-9))

    latency_ref = do_bench(run_ref, warmup=500, rep=1000)
    print("flash: {:.2f} ms".format(latency_ref))
    print("tflops: {:.2f}".format(tflops / latency_ref * 1e-9))
    return latency, (tflops / latency *
                     1e-9), latency_ref, (tflops / latency_ref * 1e-9)


def do_bench_reluattn(attn, B, H, S, D, DV,
                      dtype=torch.float16, causal=False, requires_grad=False):
    tflops = 2 * B * H * S * S * D + 2 * B * H * S * S * DV
    tflops = tflops * 0.5 if causal else tflops
    bwd_tflops = 4 * B * H * S * S * DV + 6 * B * H * S * S * D
    bwd_tflops = bwd_tflops * 0.5 if causal else bwd_tflops
    torch.cuda.manual_seed(0)
    dtype = dtype
    device = "cuda"
    accum_dtype = torch.float32
    query = torch.randn(
        B, S, H, D, device=device, dtype=dtype, requires_grad=requires_grad
    )
    key = torch.randn(
        B, S, H, D, device=device, dtype=dtype, requires_grad=requires_grad
    )
    value = torch.randn(
        B, S, H, DV, device=device, dtype=dtype, requires_grad=requires_grad
    )
    do = torch.randn(
        B, S, H, DV, device=device, dtype=dtype, requires_grad=False
    )

    o = attn(query, key, value)
    if requires_grad:
        o.backward(do, retain_graph=True)
        dQ, query.grad = query.grad.clone(), None
        dK, key.grad = key.grad.clone(), None
        dV, value.grad = value.grad.clone(), None

    def ref_program(query, key, value):
        qk = torch.einsum('bqhd,bkhd->bhqk', query, key)
        qk = qk / (D ** 0.5)
        qk = F.relu(qk)
        o = torch.einsum('bhqk,bkhd->bqhd', qk, value)
        return o

    o_ref = ref_program(query, key, value)
    print_debug(o, o_ref)
    if requires_grad:
        o_ref.backward(do, retain_graph=True)
        print_debug(query.grad, dQ)
        print_debug(key.grad, dK)
        print_debug(value.grad, dV)

    from tilelang.profiler import do_bench

    def run():
        o = attn(query, key, value)

    def run_bacward():
        o.backward(do, retain_graph=True)

    def run_ref():
        o_ref = ref_program(query, key, value)

    def run_ref_backward():
        o_ref.backward(do, retain_graph=True)

    latency = do_bench(run, warmup=500, rep=1000)
    print("tl: {:.2f} ms".format(latency))
    print("tflops: {:.2f}".format(tflops / latency * 1e-9))

    latency = do_bench(run_ref, warmup=500, rep=1000)
    print("flash: {:.2f} ms".format(latency))
    print("tflops: {:.2f}".format(tflops / latency * 1e-9))
    if requires_grad:
        latency = do_bench(run_bacward, warmup=500, rep=1000)
        print("tl bwd: {:.2f} ms".format(latency))
        print("tflops: {:.2f}".format(bwd_tflops / latency * 1e-9))

        latency = do_bench(run_ref_backward, warmup=500, rep=1000)
        print("flash bwd: {:.2f} ms".format(latency))
        print("tflops: {:.2f}".format(bwd_tflops / latency * 1e-9))


def do_bench_retention(attn, B, H, S, D, DV, dtype=torch.bfloat16):
    tflops = 2 * B * H * S * S * D + 2 * B * H * S * S * DV
    tflops = tflops * 0.5
    bwd_tflops = 4 * B * H * S * S * DV + 6 * B * H * S * S * D
    bwd_tflops = bwd_tflops * 0.5
    torch.cuda.manual_seed(0)
    dtype = dtype
    device = "cuda"
    accum_dtype = torch.float32
    query = torch.randn(
        B, S, H, D, device=device, dtype=dtype, requires_grad=True
    ) * ((1 / D)**0.5)
    key = torch.randn(
        B, S, H, D, device=device, dtype=dtype, requires_grad=True
    )
    value = torch.randn(
        B, S, H, DV, device=device, dtype=dtype, requires_grad=True
    )
    do = torch.randn(
        B, S, H, DV, device=device, dtype=dtype, requires_grad=True
    )
    mask = torch.rand(
        1, H, S, S, device="cuda", dtype=dtype, requires_grad=False
    ).tril().contiguous()
    # query = torch.empty(B, S, H, D, device=device, dtype=dtype).normal_(-0.1, 0.1)
    # key = torch.empty(B, S, H, D, device=device, dtype=dtype).normal_(-0.1, 0.1)
    # value = torch.empty(B, S, H, DV, device=device, dtype=dtype).normal_(-0.1, 0.1)
    # do = torch.empty(B, S, H, DV, device=device, dtype=dtype).normal_(-0.1, 0.1)
    # mask = torch.empty(1, H, S, S, device=device,
    # dtype=dtype).normal_(-0.1,0.1)# .tril().contiguous()

    o = attn(query, key, value, mask)
    # o.backward(do, retain_graph=True)
    # print(query.grad)
    # print(key.grad)
    # print(value.grad)

    def ref_program(query, key, value, mask):
        qk = torch.einsum('bqhd,bkhd->bhqk', query, key)
        qkm = qk * mask
        r = qkm.detach().abs().sum(dim=-1, keepdim=True).clamp(min=1.0)
        o = torch.einsum('bhqk,bkhd->bqhd', qkm / r, value)
        return o.to(dtype=dtype)
    o_ref = ref_program(query, key, value, mask)
    # print("o",o)
    # print("o_ref",o_ref)
    print_debug(o, o_ref, 1e-2, 1e-2)
    # print(o.shape)
    # print(o_ref.shape)
    # analysis_tensor_data(o, plot=True, figure_name='tensor_distribution_o.png')
    # analysis_tensor_data(o_ref, plot=True, figure_name='tensor_distribution_o_ref.png')
    # analysis_tensor_data(o-o_ref, plot=True, figure_name='tensor_distribution_o_diff.png')
    # torch.testing.assert_close(o,o_ref)

    from tilelang.profiler import do_bench

    def run():
        o = attn(query, key, value, mask)

    def run_ref():
        o = ref_program(query, key, value, mask)
    # def run_bacward():
    #     o.backward(do, retain_graph=True)

    do_bench(run)
    # do_bench(run_bacward)

    latency = do_bench(run, warmup=500, rep=1000)
    print("tl: {:.2f} ms".format(latency))
    print("tflops: {:.2f}".format(tflops / latency * 1e-9))
    latency_ref = do_bench(run_ref, warmup=500, rep=1000)
    print("torch: {:.2f} ms".format(latency_ref))
    print("tflops: {:.2f}".format(tflops / latency_ref * 1e-9))

    # latency = do_bench(run_bacward, warmup=500,rep=1000)
    # print("tl bwd: {:.2f} ms".format(latency))
    # print("tflops: {:.2f}".format(bwd_tflops/latency*1e-9))


def bench_retention_fwd(attn, B, H, S, D, DV, dtype=torch.bfloat16):
    tflops = 2 * B * H * S * S * D + 2 * B * H * S * S * DV
    tflops = tflops * 0.5
    bwd_tflops = 4 * B * H * S * S * DV + 6 * B * H * S * S * D
    bwd_tflops = bwd_tflops * 0.5
    torch.cuda.manual_seed(0)
    dtype = dtype
    device = "cuda"
    accum_dtype = torch.float32
    query = torch.randn(
        B, S, H, D, device=device, dtype=dtype, requires_grad=True
    ) * ((1 / D)**0.5)
    key = torch.randn(
        B, S, H, D, device=device, dtype=dtype, requires_grad=True
    )
    value = torch.randn(
        B, S, H, DV, device=device, dtype=dtype, requires_grad=True
    )
    do = torch.randn(
        B, S, H, DV, device=device, dtype=dtype, requires_grad=True
    )
    mask = torch.rand(
        1, H, S, S, device="cuda", dtype=dtype, requires_grad=False
    ).tril().contiguous()

    o = attn(query, key, value, mask)
    # o.backward(do, retain_graph=True)
    # print(query.grad)
    # print(key.grad)
    # print(value.grad)

    def ref_program(query, key, value, mask):
        qk = torch.einsum('bqhd,bkhd->bhqk', query, key)
        qkm = qk * mask
        r = qkm.detach().abs().sum(dim=-1, keepdim=True).clamp(min=1.0)
        o = torch.einsum('bhqk,bkhd->bqhd', qkm / r, value)
        return o.to(dtype=dtype)
    o_ref = ref_program(query, key, value, mask)
    # print("o",o)
    # print("o_ref",o_ref)
    # print_debug(o,o_ref,1e-2,1e-2)
    # torch.testing.assert_close(o,o_ref)

    from tilelang.profiler import do_bench

    def run():
        o = attn(query, key, value, mask)

    def run_ref():
        o = ref_program(query, key, value, mask)
    # def run_bacward():
    #     o.backward(do, retain_graph=True)

    do_bench(run)
    # do_bench(run_bacward)

    latency = do_bench(run, warmup=500, rep=1000)
    print("tl: {:.2f} ms".format(latency))
    print("tflops: {:.2f}".format(tflops / latency * 1e-9))
    latency_ref = do_bench(run_ref, warmup=500, rep=1000)
    print("torch: {:.2f} ms".format(latency_ref))
    print("tflops: {:.2f}".format(tflops / latency_ref * 1e-9))

    # latency = do_bench(run_bacward, warmup=500,rep=1000)
    # print("tl bwd: {:.2f} ms".format(latency))
    # print("tflops: {:.2f}".format(bwd_tflops/latency*1e-9))
    output_dict = {
        "latency": latency,
        "tflops": (tflops / latency * 1e-9),
        "latecy_ref": latency_ref,
        "tflops_ref": (tflops / latency_ref * 1e-9),
    }
    return output_dict


def do_bench_attention(attn, B, H, S, D, DV, mod=None, dtype=torch.float16,
                       seqlenq=None, require_grad=False, causal=True):
    if seqlenq is None:
        seqlenq = S
    tflops = 2 * B * H * seqlenq * S * D + 2 * B * H * seqlenq * S * DV
    tflops = tflops * 0.5 if causal else tflops
    bwd_tflops = 4 * B * H * S * S * DV + 6 * B * H * S * S * D
    bwd_tflops = bwd_tflops * 0.5 if causal else bwd_tflops
    torch.cuda.manual_seed(0)
    dtype = dtype
    device = "cuda"
    accum_dtype = torch.float32
    enable_fa3 = True
    query = torch.randn(
        B, seqlenq, H, D, device=device, dtype=dtype, requires_grad=require_grad
    )
    key = torch.randn(
        B, S, H, D, device=device, dtype=dtype, requires_grad=require_grad
    )
    value = torch.randn(
        B, S, H, DV, device=device, dtype=dtype, requires_grad=require_grad
    )
    do = torch.randn(
        B, seqlenq, H, DV, device=device, dtype=dtype, requires_grad=False
    )

    o = attn(query, key, value)
    if require_grad:
        o.backward(do, retain_graph=True)
        dQ, query.grad = query.grad.clone(), None
        dK, key.grad = key.grad.clone(), None
        dV, value.grad = value.grad.clone(), None

    try:
        from flash_attn_interface import flash_attn_func as flash_attn_func_hopper
        # from flash_attn import flash_attn_func as flash_attn_func_hopper
    except BaseException:
        flash_attn_func_hopper = None
        enable_fa3 = False

    DIM_HOPPER = [64, 128, 256]
    dim_padded_fa3 = list(filter(lambda x: x >= max(D, DV), DIM_HOPPER))
    assert (len(dim_padded_fa3) > 0)
    dim_padded_fa3 = min(dim_padded_fa3)

    def fa3(dim_padded):
        if D < dim_padded:
            query_padded = F.pad(query, (0, dim_padded - D), value=0.)
            key_padded = F.pad(key, (0, dim_padded - D), value=0.)
        else:
            query_padded = query
            key_padded = key
        if DV < dim_padded:
            value_padded = F.pad(value, (0, dim_padded - DV), value=0.)
        else:
            value_padded = value
        o_ref = flash_attn_func_hopper(
            query_padded, key_padded, value_padded, softmax_scale=(
                1 / D)**0.5, causal=causal)[0]
        if DV < dim_padded:
            o_ref = o_ref[:, :, :, :DV]
        return o_ref

    # o_ref = fa3(dim_padded_fa3)
    # print_debug(o,o_ref)

    from flash_attn import flash_attn_func
    dim_padded_fa2 = max(D, DV)

    def fa2(query, key, value, dim_padded):
        if D < dim_padded:
            query_padded = F.pad(query, (0, dim_padded - D), value=0.)
            key_padded = F.pad(key, (0, dim_padded - D), value=0.)
        else:
            query_padded = query
            key_padded = key
        if DV < dim_padded:
            value_padded = F.pad(value, (0, dim_padded - DV), value=0.)
        else:
            value_padded = value
        o_ref = flash_attn_func(
            query_padded,
            key_padded,
            value_padded,
            softmax_scale=(
                1 / D)**0.5,
            causal=causal)
        if DV < dim_padded:
            o_ref = o_ref[:, :, :, :DV]
        return o_ref
    o_ref = fa2(query, key, value, dim_padded_fa2)
    print_debug(o, o_ref)
    if require_grad:
        o_ref.backward(do, retain_graph=True)
        print_debug(query.grad, dQ)
        print_debug(key.grad, dK)
        print_debug(value.grad, dV)

    # from tilelang.profiler import do_bench
    def run():
        o = attn(query, key, value)

    def run_ref():
        fa2(query, key, value, dim_padded_fa2)

    def run_ref_fa3():
        fa3(dim_padded_fa3)

    def run_bacward():
        o.backward(do, retain_graph=True)

    def run_bacward_ref():
        o_ref.backward(do, retain_graph=True)
    o_reffa3 = fa3(dim_padded_fa3) if enable_fa3 else None

    def run_bacward_ref_fa3():
        o_reffa3.backward(do, retain_graph=True)

    # do_bench(run)
    # do_bench(run_bacward)
    import tilelang as tl
    if mod:
        program = mod(B, H, S, D, DV, 64, 64, 2, 128)
        mod, params = tl.lower(program)
        mod = tl.Profiler(mod, params, [3, 4], tl.TensorSupplyType.Normal)
        latecy = mod.do_bench(mod, n_warmup=10, n_repeat=10, profiler="torch")
        print("tl profile: {:.2f} ms".format(latecy))
        print("tflops: {:.2f}".format(tflops / latecy * 1e-9))

    # tl slow down when rep too large
    latency = do_bench(run, warmup=50, rep=100)
    print("tl: {:.2f} ms".format(latency))
    print("tflops: {:.2f}".format(tflops / latency * 1e-9))
    latency_ref = do_bench(run_ref, warmup=50, rep=100)
    print("flash: {:.2f} ms".format(latency_ref))
    print("tflops: {:.2f}".format(tflops / latency_ref * 1e-9))
    if enable_fa3:
        latency_reffa3 = do_bench(run_ref_fa3, warmup=50, rep=100)
        print("flash fa3: {:.2f} ms".format(latency_reffa3))
        print("tflops: {:.2f}".format(tflops / latency_reffa3 * 1e-9))
    if require_grad:
        latency = do_bench(run_bacward, warmup=50, rep=100)
        print("tl bwd: {:.2f} ms".format(latency))
        print("tflops: {:.2f}".format(bwd_tflops / latency * 1e-9))
        latency_ref = do_bench(run_bacward_ref, warmup=50, rep=100)
        print("flash bwd: {:.2f} ms".format(latency_ref))
        print("tflops: {:.2f}".format(bwd_tflops / latency_ref * 1e-9))
        if enable_fa3 and dim_padded_fa3 <= 128:
            latency_reffa3 = do_bench(run_bacward_ref_fa3, warmup=50, rep=100)
            print("flash fa3 bwd: {:.2f} ms".format(latency_reffa3))
            print("tflops: {:.2f}".format(bwd_tflops / latency_reffa3 * 1e-9))


def do_bench_attention_bwd_fa2(attn, B, H, S, D, DV, mod=None,
                               dtype=torch.float16, seqlenq=None, require_grad=False, causal=True):
    if seqlenq is None:
        seqlenq = S
    tflops = 2 * B * H * seqlenq * S * D + 2 * B * H * seqlenq * S * DV
    tflops = tflops * 0.5 if causal else tflops
    bwd_tflops = 4 * B * H * S * S * DV + 6 * B * H * S * S * D
    bwd_tflops = bwd_tflops * 0.5 if causal else bwd_tflops
    torch.cuda.manual_seed(0)
    dtype = dtype
    device = "cuda"
    accum_dtype = torch.float32
    query = torch.randn(
        B, seqlenq, H, D, device=device, dtype=dtype, requires_grad=require_grad
    )
    key = torch.randn(
        B, S, H, D, device=device, dtype=dtype, requires_grad=require_grad
    )
    value = torch.randn(
        B, S, H, DV, device=device, dtype=dtype, requires_grad=require_grad
    )
    do = torch.randn(
        B, seqlenq, H, DV, device=device, dtype=dtype, requires_grad=False
    )

    o = attn(query, key, value)
    if require_grad:
        o.backward(do, retain_graph=True)
        dQ, query.grad = query.grad.clone(), None
        dK, key.grad = key.grad.clone(), None
        dV, value.grad = value.grad.clone(), None

    from flash_attn import flash_attn_func
    dim_padded_fa2 = max(D, DV)

    def fa2(query, key, value, dim_padded):
        if D < dim_padded:
            query_padded = F.pad(query, (0, dim_padded - D), value=0.)
            key_padded = F.pad(key, (0, dim_padded - D), value=0.)
        else:
            query_padded = query
            key_padded = key
        if DV < dim_padded:
            value_padded = F.pad(value, (0, dim_padded - DV), value=0.)
        else:
            value_padded = value
        o_ref = flash_attn_func(
            query_padded,
            key_padded,
            value_padded,
            softmax_scale=(
                1 / D)**0.5,
            causal=causal)
        if DV < dim_padded:
            o_ref = o_ref[:, :, :, :DV]
        return o_ref
    o_ref = fa2(query, key, value, dim_padded_fa2)
    print_debug(o, o_ref)
    if require_grad:
        o_ref.backward(do, retain_graph=True)
        print_debug(query.grad, dQ)
        print_debug(key.grad, dK)
        print_debug(value.grad, dV)

    from tilelang.profiler import do_bench

    def run():
        o = attn(query, key, value)

    def run_ref():
        fa2(query, key, value, dim_padded_fa2)

    def run_bacward():
        o.backward(do, retain_graph=True)

    def run_bacward_ref():
        o_ref.backward(do, retain_graph=True)

    # tl slow down when rep too large
    # latency = do_bench(run, warmup=50,rep=100)
    # print("tl: {:.2f} ms".format(latency))
    # print("tflops: {:.2f}".format(tflops/latency*1e-9))
    # latency_ref = do_bench(run_ref, warmup=50,rep=100)
    # print("flash: {:.2f} ms".format(latency_ref))
    # print("tflops: {:.2f}".format(tflops/latency_ref*1e-9))
    if require_grad:
        latency = do_bench(run_bacward, warmup=50, rep=100)
        print("tl bwd: {:.2f} ms".format(latency))
        print("tflops: {:.2f}".format(bwd_tflops / latency * 1e-9))
        latency_ref = do_bench(run_bacward_ref, warmup=50, rep=100)
        print("flash bwd: {:.2f} ms".format(latency_ref))
        print("tflops: {:.2f}".format(bwd_tflops / latency_ref * 1e-9))


def bench_attention_fwd(attn, B, H, S, D, DV):
    tflops = 2 * B * H * S * S * D + 2 * B * H * S * S * DV
    tflops = tflops * 0.5
    bwd_tflops = 4 * B * H * S * S * DV + 6 * B * H * S * S * D
    bwd_tflops = bwd_tflops * 0.5
    torch.cuda.manual_seed(0)
    dtype = torch.float16
    device = "cuda"
    accum_dtype = torch.float32
    query = torch.randn(
        B, S, H, D, device=device, dtype=dtype, requires_grad=True
    )
    key = torch.randn(
        B, S, H, D, device=device, dtype=dtype, requires_grad=True
    )
    value = torch.randn(
        B, S, H, DV, device=device, dtype=dtype, requires_grad=True
    )
    do = torch.randn(
        B, S, H, DV, device=device, dtype=dtype, requires_grad=True
    )

    o = attn(query, key, value)
    # print(o)
    # o.backward(do, retain_graph=True)
    # print(query.grad)
    # print(key.grad)
    # print(value.grad)

    from flash_attn_interface import flash_attn_func as flash_attn_func_hopper

    DIM_HOPPER = [64, 128, 256]
    dim_padded_fa3 = list(filter(lambda x: x >= max(D, DV), DIM_HOPPER))
    assert (len(dim_padded_fa3) > 0)
    dim_padded_fa3 = min(dim_padded_fa3)

    def fa3(dim_padded):
        if D < dim_padded:
            query_padded = F.pad(query, (0, dim_padded - D), value=0.)
            key_padded = F.pad(key, (0, dim_padded - D), value=0.)
        else:
            query_padded = query
            key_padded = key
        if DV < dim_padded:
            value_padded = F.pad(value, (0, dim_padded - DV), value=0.)
        else:
            value_padded = value
        o_ref = flash_attn_func_hopper(
            query_padded,
            key_padded,
            value_padded,
            softmax_scale=(
                1 / D)**0.5,
            causal=True)[0]
        if DV < dim_padded:
            o_ref = o_ref[:, :, :, :DV]
        return o_ref

    o_ref = fa3(dim_padded_fa3)
    # print_debug(o,o_ref)

    from flash_attn import flash_attn_func
    dim_padded_fa2 = max(D, DV)

    def fa2(dim_padded):
        if D < dim_padded:
            query_padded = F.pad(query, (0, dim_padded - D), value=0.)
            key_padded = F.pad(key, (0, dim_padded - D), value=0.)
        else:
            query_padded = query
            key_padded = key
        if DV < dim_padded:
            value_padded = F.pad(value, (0, dim_padded - DV), value=0.)
        else:
            value_padded = value
        o_ref = flash_attn_func(
            query_padded,
            key_padded,
            value_padded,
            softmax_scale=(
                1 / D)**0.5,
            causal=True)
        if DV < dim_padded:
            o_ref = o_ref[:, :, :, :DV]
        return o_ref
    o_ref = fa2(dim_padded_fa2)
    # print_debug(o,o_ref)

    from tilelang.profiler import do_bench

    def run():
        o = attn(query, key, value)

    def run_ref():
        fa2(dim_padded_fa2)

    def run_ref_fa3():
        fa3(dim_padded_fa3)

    def run_bacward():
        o.backward(do, retain_graph=True)

    # do_bench(run)
    # do_bench(run_bacward)

    latency = do_bench(run, warmup=500, rep=1000)
    print("tl: {:.2f} ms".format(latency))
    print("tflops: {:.2f}".format(tflops / latency * 1e-9))
    latency_ref = do_bench(run_ref, warmup=500, rep=1000)
    print("flash: {:.2f} ms".format(latency_ref))
    print("tflops: {:.2f}".format(tflops / latency_ref * 1e-9))
    latency_reffa3 = do_bench(run_ref_fa3, warmup=500, rep=1000)
    print("flash fa3: {:.2f} ms".format(latency_reffa3))
    print("tflops: {:.2f}".format(tflops / latency_reffa3 * 1e-9))

    # latency = do_bench(run_bacward, warmup=500,rep=1000)
    # print("tl bwd: {:.2f} ms".format(latency))
    # print("tflops: {:.2f}".format(bwd_tflops/latency*1e-9))

    output_dict = {
        "latency": latency,
        "tflops": (tflops / latency * 1e-9),
        "latecy_ref": latency_ref,
        "tflops_ref": (tflops / latency_ref * 1e-9),
        "latecy_reffa3": latency_reffa3,
        "tflops_reffa3": (tflops / latency_reffa3 * 1e-9),
    }
    return output_dict


def do_bench_attention_128256(B, H, S, D, DV, dtype=torch.float16):
    tflops = 2 * B * H * S * S * D + 2 * B * H * S * S * DV
    tflops = tflops * 0.5
    bwd_tflops = 4 * B * H * S * S * DV + 6 * B * H * S * S * D
    bwd_tflops = bwd_tflops * 0.5
    torch.cuda.manual_seed(0)
    dtype = dtype
    device = "cuda"
    accum_dtype = torch.float32
    require_grad = True
    query = torch.randn(
        B, S, H, D, device=device, dtype=dtype, requires_grad=require_grad
    )
    key = torch.randn(
        B, S, H, D, device=device, dtype=dtype, requires_grad=require_grad
    )
    value = torch.randn(
        B, S, H, DV, device=device, dtype=dtype, requires_grad=require_grad
    )
    do = torch.randn(
        B, S, H, DV, device=device, dtype=dtype, requires_grad=False
    )

    from flash_attn_interface import flash_attn_func as flash_attn_func_hopper
    # from flash_attn import flash_attn_func as flash_attn_func_hopper

    DIM_HOPPER = [64, 128, 256]
    assert (D == 128 and DV == 256)

    def fa3_split():
        query_padded = query
        key_padded = key
        value1, value2 = torch.split(value, 128, dim=-1)
        o_ref1 = flash_attn_func_hopper(
            query_padded, key_padded, value1, softmax_scale=(
                1 / D)**0.5, causal=True)[0]
        o_ref2 = flash_attn_func_hopper(
            query_padded, key_padded, value2, softmax_scale=(
                1 / D)**0.5, causal=True)[0]
        o_ref = torch.cat([o_ref1, o_ref2], dim=-1)
        return o_ref

    o_ref3 = fa3_split()

    from flash_attn import flash_attn_func
    dim_padded_fa2 = max(D, DV)

    def fa2(dim_padded):
        if D < dim_padded:
            query_padded = F.pad(query, (0, dim_padded - D), value=0.)
            key_padded = F.pad(key, (0, dim_padded - D), value=0.)
        else:
            query_padded = query
            key_padded = key
        if DV < dim_padded:
            value_padded = F.pad(value, (0, dim_padded - DV), value=0.)
        else:
            value_padded = value
        o_ref = flash_attn_func(
            query_padded,
            key_padded,
            value_padded,
            softmax_scale=(
                1 / D)**0.5,
            causal=True)
        if DV < dim_padded:
            o_ref = o_ref[:, :, :, :DV]
        return o_ref
    o_ref = fa2(dim_padded_fa2)
    print_debug(o_ref, o_ref3)

    # from tilelang.profiler import do_bench
    def run_ref():
        fa2(dim_padded_fa2)

    def run_ref_fa3():
        fa3_split()

    # tl slow down when rep too large
    latency_ref = do_bench(run_ref, warmup=50, rep=100)
    print("flash: {:.2f} ms".format(latency_ref))
    print("tflops: {:.2f}".format(tflops / latency_ref * 1e-9))
    latency_reffa3 = do_bench(run_ref_fa3, warmup=50, rep=100)
    print("flash fa3: {:.2f} ms".format(latency_reffa3))
    print("tflops: {:.2f}".format(tflops / latency_reffa3 * 1e-9))

    # latency = do_bench(run_bacward, warmup=500,rep=1000)
    # print("tl bwd: {:.2f} ms".format(latency))
    # print("tflops: {:.2f}".format(bwd_tflops/latency*1e-9))


def do_bench_flex_attention(
        attn, B, H, S, D, DV, dtype=torch.float16, require_grad=False):
    tflops = 2 * B * H * S * S * D + 2 * B * H * S * S * DV
    tflops = tflops * 0.5
    bwd_tflops = 4 * B * H * S * S * DV + 6 * B * H * S * S * D
    bwd_tflops = bwd_tflops * 0.5
    torch.cuda.manual_seed(0)
    dtype = dtype
    device = "cuda"
    accum_dtype = torch.float32
    require_grad = require_grad
    query = torch.randn(
        B, H, S, D, device=device, dtype=dtype, requires_grad=require_grad
    )
    key = torch.randn(
        B, H, S, D, device=device, dtype=dtype, requires_grad=require_grad
    )
    value = torch.randn(
        B, H, S, DV, device=device, dtype=dtype, requires_grad=require_grad
    )
    do = torch.randn(
        B, H, S, DV, device=device, dtype=dtype, requires_grad=False
    )

    from torch.nn.attention.flex_attention import (
        _DEFAULT_SPARSE_BLOCK_SIZE,
        create_block_mask,
        create_mask,
        flex_attention,
    )

    def causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    @lru_cache
    def create_block_mask_cached(score_mod, B, H, M, N, device="cuda"):
        block_mask = create_block_mask(score_mod, B, H, M, N, device=device)
        return block_mask

    block_mask = create_block_mask_cached(
        causal_mask, 1, 1, S, S, device=query.device)
    flex_attention = torch.compile(flex_attention, dynamic=False)
    # run = lambda: flex_attention(
    #     query, key, value, block_mask=block_mask
    # )
    DIM_HOPPER = [64, 128, 256]
    dim_padded_fa3 = list(filter(lambda x: x >= max(D, DV), DIM_HOPPER))
    assert (len(dim_padded_fa3) > 0)
    dim_padded_fa3 = min(dim_padded_fa3)

    def fa3(dim_padded):
        if D < dim_padded:
            query_padded = F.pad(query, (0, dim_padded - D), value=0.)
            key_padded = F.pad(key, (0, dim_padded - D), value=0.)
        else:
            query_padded = query
            key_padded = key
        if DV < dim_padded:
            value_padded = F.pad(value, (0, dim_padded - DV), value=0.)
        else:
            value_padded = value
        o_ref = flex_attention(
            query_padded,
            key_padded,
            value_padded,
            block_mask=block_mask)
        if DV < dim_padded:
            o_ref = o_ref[:, :, :, :DV]
        return o_ref

    o_ref = fa3(dim_padded_fa3)

    def run(): return fa3(dim_padded_fa3)

    def run_bacward():
        o_ref.backward(do, retain_graph=True)

    # tl slow down when rep too large
    latency = do_bench(run, warmup=50, rep=100)
    print("tl: {:.2f} ms".format(latency))
    print("tflops: {:.2f}".format(tflops / latency * 1e-9))

    if require_grad:
        latency = do_bench(run_bacward, warmup=50, rep=100)
        print("tl bwd: {:.2f} ms".format(latency))
        print("tflops: {:.2f}".format(bwd_tflops / latency * 1e-9))


def do_bench_flashinfer(attn, B, H, S, D, DV, dtype=torch.float16):
    # assert(B==1)
    tflops = 2 * B * H * S * S * D + 2 * B * H * S * S * DV
    tflops = tflops * 0.5
    bwd_tflops = 4 * B * H * S * S * DV + 6 * B * H * S * S * D
    bwd_tflops = bwd_tflops * 0.5
    torch.cuda.manual_seed(0)
    dtype = dtype
    device = "cuda"
    accum_dtype = torch.float32
    require_grad = False

    import flashinfer
    from flashinfer import prefill
    DIM_HOPPER = [64, 128, 256]
    dim_padded_fa3 = list(filter(lambda x: x >= max(D, DV), DIM_HOPPER))
    assert (len(dim_padded_fa3) > 0)
    dim_padded_fa3 = min(dim_padded_fa3)

    num_layers = 1  # 32
    num_qo_heads = H  # 64
    num_kv_heads = H  # 16
    head_dim = dim_padded_fa3  # 128
    page_size = 16
    max_num_pages = B * S // page_size  # 128

    if B == 1:
        query = torch.randn(
            B * S, H, D, device=device, dtype=dtype, requires_grad=require_grad
        )
        key = torch.randn(
            B * S, H, D, device=device, dtype=dtype, requires_grad=require_grad
        )
        value = torch.randn(
            B * S, H, DV, device=device, dtype=dtype, requires_grad=require_grad
        )
    elif B != 1:
        query = torch.randn(
            B * S, H, dim_padded_fa3, device=device, dtype=dtype, requires_grad=require_grad
        )
        kv_cache = torch.randn(
            max_num_pages, 2, page_size, H, dim_padded_fa3, device=device, dtype=dtype, requires_grad=require_grad
        )

    # out = prefill.single_prefill_with_kv_cache(query, key, value, causal=True)
    # run = lambda: prefill.single_prefill_with_kv_cache(
    #     query, key, value, causal=True
    # )
    # allocate 128MB workspace buffer
    if B != 1:
        workspace_buffer = torch.empty(
            128 * 1024 * 1024, dtype=torch.uint8, device=device)
        prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            workspace_buffer, "NHD"
        )
        #     qo_indptr = torch.tensor(
        #     # [0, 33, 44, 55, 66, 77, 88, nnz_qo], dtype=torch.int32, device="cuda:0"
        # )
        qo_indptr = torch.range(0, S * B, S, dtype=torch.int32, device=device)
        paged_kv_indices = torch.arange(max_num_pages).int().to(device)
        # paged_kv_indptr = torch.tensor(
        #     [0, 17, 29, 44, 48, 66, 100, 128], dtype=torch.int32, device="cuda:0"
        # )
        paged_kv_indptr = torch.range(
            0,
            max_num_pages,
            S // page_size,
            dtype=torch.int32,
            device=device)
        # 1 <= paged_kv_last_page_len <= page_size
        # paged_kv_last_page_len = torch.tensor(
        #     [1, 7, 14, 4, 3, 1, 16], dtype=torch.int32, device="cuda:0"
        # )
        paged_kv_last_page_len = torch.tensor(
            [page_size] * B, dtype=torch.int32, device=device
        )
        # create auxiliary data structures for batch prefill attention
        prefill_wrapper.plan(
            qo_indptr,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            causal=True,
        )
    # o = prefill_wrapper.run(query, kv_cache)

    def fa3(dim_padded):
        if D < dim_padded:
            query_padded = F.pad(query, (0, dim_padded - D), value=0.)
            key_padded = F.pad(key, (0, dim_padded - D), value=0.)
        else:
            query_padded = query
            key_padded = key
        if DV < dim_padded:
            value_padded = F.pad(value, (0, dim_padded - DV), value=0.)
        else:
            value_padded = value
        o_ref = prefill.single_prefill_with_kv_cache(
            query_padded, key_padded, value_padded, causal=True)
        if DV < dim_padded:
            o_ref = o_ref[:, :, :DV]
        return o_ref

    if B == 1:
        def run(): return fa3(dim_padded_fa3)
    else:
        def run(): return prefill_wrapper.run(query, kv_cache)

    # from tilelang.profiler import do_bench

    # tl slow down when rep too large
    latency = do_bench(run, warmup=50, rep=100)
    print("tl: {:.2f} ms".format(latency))
    print("tflops: {:.2f}".format(tflops / latency * 1e-9))


def do_bench_sigmoid_flashinfer(attn, B, H, S, D, DV, dtype=torch.float16):
    # assert(B==1)
    tflops = 2 * B * H * S * S * D + 2 * B * H * S * S * DV
    tflops = tflops * 0.5
    bwd_tflops = 4 * B * H * S * S * DV + 6 * B * H * S * S * D
    bwd_tflops = bwd_tflops * 0.5
    torch.cuda.manual_seed(0)
    dtype = dtype
    device = "cuda"
    accum_dtype = torch.float32
    require_grad = False

    import flashinfer
    import flashinfer.jit
    from flashinfer import prefill
    DIM_HOPPER = [64, 128, 256]
    dim_padded_fa3 = list(filter(lambda x: x >= max(D, DV), DIM_HOPPER))
    assert (len(dim_padded_fa3) > 0)
    dim_padded_fa3 = min(dim_padded_fa3)

    from flashinfer.jit.attention import (
        gen_customize_single_prefill_module
        # gen_customize_single_prefill_sm90_module as
        # gen_customize_single_prefill_module
    )
    from flashinfer.prefill import single_prefill_with_kv_cache_with_jit_module
    from flashinfer.utils import MaskMode

    num_layers = 1  # 32
    num_qo_heads = H  # 64
    num_kv_heads = H  # 16
    head_dim = dim_padded_fa3  # 128
    page_size = 16
    max_num_pages = B * S // page_size  # 128

    variant_decl = r"""
    template <typename ParamsT_>
    struct FlashSigmoid {
    using ParamsT = ParamsT_;
    using DTypeQ = typename ParamsT::DTypeQ;
    using DTypeKV = typename ParamsT::DTypeKV;
    using DTypeO = typename ParamsT::DTypeO;
    using IdType = typename ParamsT::IdType;
    static constexpr bool use_softmax = false;

    uint32_t window_left, qo_len, kv_len;
    float sigmoid_bias_log2e;

    // Create closure
    __device__ __host__ FlashSigmoid(const ParamsT& params, uint32_t batch_idx,
                                    uint8_t* smem_ptr) {
        qo_len = params.get_qo_len(batch_idx);
        kv_len = params.get_kv_len(batch_idx);
        window_left = kv_len;
        sigmoid_bias_log2e = params.sigmoid_bias * math::log2e;
    }

    template <typename T>
    __device__ __forceinline__ T QueryTransform(const ParamsT& params, T q) {
        return float(q) * params.logits_scale * math::log2e;
    }

    template <typename T>
    __device__ __forceinline__ T LogitsTransform(const ParamsT& params, T logits, uint32_t batch_idx,
                                                uint32_t qo_idx, uint32_t kv_idx,
                                                uint32_t qo_head_idx, uint32_t kv_head_idx) {
        return math::ptx_rcp(1.f + math::ptx_exp2(-float(logits + sigmoid_bias_log2e)));
    }

    __device__ __forceinline__ bool LogitsMask(const ParamsT& params, uint32_t batch_idx,
                                                uint32_t qo_idx, uint32_t kv_idx, uint32_t qo_head_idx,
                                                uint32_t kv_head_idx) {
        return true;
    }
    };
    """
    jit_module = gen_customize_single_prefill_module(
        "flash_sigmoid",
        dtype,  # dtype_q
        dtype,  # dtype_kv
        dtype,  # dtype_o
        head_dim,  # hidden_dim
        [],  # additional_input_tensor_var_names
        [],  # additional_input_tensor_var_types
        ["logits_scale", "sigmoid_bias"],  # additional_input_scalar_var_names
        ["float", "float"],  # additional_input_scalar_var_types
        "FlashSigmoid",
        variant_decl,
    )
    f = functools.partial(
        single_prefill_with_kv_cache_with_jit_module,
        jit_module)
    assert (B == 1)

    if B == 1:
        query = torch.randn(
            B * S, H, D, device=device, dtype=dtype, requires_grad=require_grad
        )
        key = torch.randn(
            B * S, H, D, device=device, dtype=dtype, requires_grad=require_grad
        )
        value = torch.randn(
            B * S, H, DV, device=device, dtype=dtype, requires_grad=require_grad
        )
    elif B != 1:
        query = torch.randn(
            B * S, H, dim_padded_fa3, device=device, dtype=dtype, requires_grad=require_grad
        )
        kv_cache = torch.randn(
            max_num_pages, 2, page_size, H, dim_padded_fa3, device=device, dtype=dtype, requires_grad=require_grad
        )
    logits_scale = 1.0 / math.sqrt(head_dim)
    sigmoid_bias = 0.25
    o = f(
        query,
        key,
        value,
        logits_scale,
        sigmoid_bias,
        mask_mode=MaskMode.NON_CAUSAL.value)

    # out = prefill.single_prefill_with_kv_cache(query, key, value, causal=True)
    # run = lambda: prefill.single_prefill_with_kv_cache(
    #     query, key, value, causal=True
    # )
    # allocate 128MB workspace buffer
    if B != 1:
        workspace_buffer = torch.empty(
            128 * 1024 * 1024, dtype=torch.uint8, device=device)
        prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            workspace_buffer, "NHD"
        )
        #     qo_indptr = torch.tensor(
        #     # [0, 33, 44, 55, 66, 77, 88, nnz_qo], dtype=torch.int32, device="cuda:0"
        # )
        qo_indptr = torch.range(0, S * B, S, dtype=torch.int32, device=device)
        paged_kv_indices = torch.arange(max_num_pages).int().to(device)
        # paged_kv_indptr = torch.tensor(
        #     [0, 17, 29, 44, 48, 66, 100, 128], dtype=torch.int32, device="cuda:0"
        # )
        paged_kv_indptr = torch.range(
            0,
            max_num_pages,
            S // page_size,
            dtype=torch.int32,
            device=device)
        # 1 <= paged_kv_last_page_len <= page_size
        # paged_kv_last_page_len = torch.tensor(
        #     [1, 7, 14, 4, 3, 1, 16], dtype=torch.int32, device="cuda:0"
        # )
        paged_kv_last_page_len = torch.tensor(
            [page_size] * B, dtype=torch.int32, device=device
        )
        # create auxiliary data structures for batch prefill attention
        prefill_wrapper.plan(
            qo_indptr,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            causal=True,
        )
    # o = prefill_wrapper.run(query, kv_cache)

    def fa3(dim_padded):
        if D < dim_padded:
            query_padded = F.pad(query, (0, dim_padded - D), value=0.)
            key_padded = F.pad(key, (0, dim_padded - D), value=0.)
        else:
            query_padded = query
            key_padded = key
        if DV < dim_padded:
            value_padded = F.pad(value, (0, dim_padded - DV), value=0.)
        else:
            value_padded = value
        o_ref = f(
            query_padded,
            key_padded,
            value_padded,
            logits_scale,
            sigmoid_bias,
            mask_mode=MaskMode.NON_CAUSAL.value)
        if DV < dim_padded:
            o_ref = o_ref[:, :, :DV]
        return o_ref

    if B == 1:
        def run(): return fa3(dim_padded_fa3)
    else:
        def run(): return prefill_wrapper.run(query, kv_cache)

    # from tilelang.profiler import do_bench

    # tl slow down when rep too large
    latency = do_bench(run, warmup=50, rep=100)
    print("tl: {:.2f} ms".format(latency))
    print("tflops: {:.2f}".format(tflops / latency * 1e-9))


#     from tilelang.profiler import do_bench
#     def run():
#         o = attn(query, key, value, softmax_bias)
#     def run_bacward():
#         o.backward(do, retain_graph=True)

#     def run_ref():
#         o_ref = flash_attn_func(query, key, value, softmax_scale=1.0,causal=True, sigmoid_bias=softmax_bias_cpu)

#     # do_bench(run)
#     # do_bench(run_bacward)

#     latency = do_bench(run, warmup=500,rep=1000)
#     print("tl: {:.2f} ms".format(latency))
#     print("tflops: {:.2f}".format(tflops/latency*1e-9))

#     # latency = do_bench(run_bacward, warmup=500,rep=1000)
#     # print("tl bwd: {:.2f} ms".format(latency))
#     # print("tflops: {:.2f}".format(bwd_tflops/latency*1e-9))

#     # do_bench(run_ref)

#     latency = do_bench(run_ref, warmup=500,rep=1000)
#     print("flash: {:.2f} ms".format(latency))
#     print("tflops: {:.2f}".format(tflops/latency*1e-9))
