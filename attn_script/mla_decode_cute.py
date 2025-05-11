from attn_engine import AttentionEngine
import torch
import math
from attn_engine import OnlineFunc
from core import CustomIO
from core import SymbolScalar
from core import Var
from core import meta_tensor

"""
Example of mla attention decode with online softmax
"""


D = 576 # 384+64 # 576
softmax_scale = 1/D ** 0.5
# elementwise on attention scores
def score_mod(score, custom_fwd_inputs, b, h, q_idx, kv_idx):
    return score * softmax_scale

class OnlineSoftmax(OnlineFunc):
    def __init__(self):
        """
        define online_rowscales and final_rowscales
        """
        online_rowscales = {
            "m": SymbolScalar("m", Var("-inf")),
            "r": SymbolScalar("r", Var("0.0")),
        }
        final_rowscales = {
            "lse": SymbolScalar("lse", Var("0.0")),
        }
        external_fwd_inputs = CustomIO()
        super().__init__(online_rowscales, final_rowscales,
                    external_fwd_inputs)
    

    # scan
    @staticmethod
    def online_fwd(scores, online_rowscales, b, h, q_idx):

        m , r = online_rowscales["m"], online_rowscales["r"]
        m_new = m.max(scores.get_reduce("max"))
        scale_tmp = (m - m_new).exp()
        r = r * scale_tmp
        
        scores = (scores - m_new).exp()
        r = r + scores.get_reduce("sum")

        new_online_rowscales = {
            "m": m_new,
            "r": r,
        }
        o_scale = scale_tmp
        return scores, new_online_rowscales, o_scale
    
    @staticmethod
    def combine(final_rowscales, ):
        lse = final_rowscales["lse"]
        lse_max = lse.get_reduce("max")
        row_sum = (lse - lse_max).exp()
        row_sum_sum = row_sum.get_reduce("sum")
        lse_sum = row_sum_sum.log() + lse_max
        o_scale = (lse - lse_sum).exp()
        return o_scale

    @staticmethod
    def online_fwd_epilogue(o, online_rowscales, b, h, q_idx):
        o_new = o / online_rowscales["r"]
        lse = (online_rowscales["r"]).log() + online_rowscales["m"]
        final_rowscales = {
            "lse": lse,
        }
        return o_new, final_rowscales

    @staticmethod
    def forward(scores, final_rowscales, b, h, q_idx, kv_idx):
        lse = final_rowscales["lse"]
        scores_new = (scores-lse).exp()
        return scores_new
    
    @staticmethod
    def backward(dp, scores, final_rowscales, doosum_rowscales, b, h, q_idx, kv_idx):
        dppsum = doosum_rowscales
        dscores = (dp - dppsum)*scores 
        return dscores

def scaled_dot_product_attention(query, key, value, h_q, h_kv, is_causal=False):
    query = query.float()
    key = key.float()
    value = value.float()
    key = key.repeat_interleave(h_q // h_kv, dim=0)
    value = value.repeat_interleave(h_q // h_kv, dim=0)
    attn_weight = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
    if is_causal:
        s_q = query.shape[-2]
        s_k = key.shape[-2]
        attn_bias = torch.zeros(s_q, s_k, dtype=query.dtype)
        temp_mask = torch.ones(s_q, s_k, dtype=torch.bool).tril(diagonal=s_k - s_q)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)
        attn_weight += attn_bias
    lse = attn_weight.logsumexp(dim=-1)
    attn_weight = torch.softmax(attn_weight, dim=-1, dtype=torch.float32)
    return attn_weight @ value, lse

def cal_diff(x: torch.Tensor, y: torch.Tensor, name: str) -> None:
    x, y = x.double(), y.double()
    RMSE = ((x - y) * (x - y)).mean().sqrt().item()
    cos_diff = 1 - 2 * (x * y).sum().item() / max((x * x + y * y).sum().item(), 1e-12)
    amax_diff = (x - y).abs().max().item()
    # print(f"{name}: {cos_diff=}, {RMSE=}, {amax_diff=}")
    assert cos_diff < 1e-5
    
@torch.inference_mode()
def test_flash_mla(b, s_q, mean_sk, h_q, h_kv, d, dv, causal, varlen):
    from flash_mla import flash_mla_with_kvcache, get_mla_metadata
    import triton
    import tilelang
    # print(
    #     f"{b=}, {s_q=}, {mean_sk=}, {h_q=}, {h_kv=}, {d=}, {dv=}, {causal=}, {varlen=}"
    # )

    cache_seqlens = torch.full((b,), mean_sk, dtype=torch.int32, device="cuda")
    # if varlen:
    #     for i in range(b):
    #         cache_seqlens[i] = max(random.normalvariate(mean_sk, mean_sk / 2), s_q)
    # total_seqlens = cache_seqlens.sum().item()
    # mean_seqlens = cache_seqlens.float().mean().int().item()
    max_seqlen = cache_seqlens.max().item()
    max_seqlen_pad = triton.cdiv(max_seqlen, 256) * 256
    # print(f"{total_seqlens=}, {mean_seqlens=}, {max_seqlen=}")

    tile_scheduler_metadata, num_splits = get_mla_metadata(
        cache_seqlens, s_q * h_q // h_kv, h_kv
    )
    block_size = 64
    block_table = torch.arange(
        b * max_seqlen_pad // block_size, dtype=torch.int32, device="cuda"
    ).view(b, max_seqlen_pad // block_size)
    
    q = torch.randn(b, s_q, h_q, d, dtype=torch.float16, device="cuda")
    blocked_k = torch.randn(block_table.numel(), block_size, h_kv, d, dtype=torch.float16, device="cuda")
    # for i in range(b):
    #     blocked_k.view(b, max_seqlen_pad, h_kv, d)[i, cache_seqlens[i].item():] = (
    #         float("nan")
    #     )
    # blocked_v = blocked_k[..., :dv]


    def flash_mla():
        return flash_mla_with_kvcache(
            q,
            blocked_k,
            block_table,
            cache_seqlens,
            dv,
            tile_scheduler_metadata,
            num_splits,
            causal=causal,
        )

    def ref_mla():
        out = torch.empty(b, s_q, h_q, dv, dtype=torch.float32)
        lse = torch.empty(b, h_q, s_q, dtype=torch.float32)
        for i in range(b):
            begin = i * max_seqlen_pad
            end = begin + cache_seqlens[i]
            O, LSE = scaled_dot_product_attention(
                q[i].transpose(0, 1),
                blocked_k.view(-1, h_kv, d)[begin:end].transpose(0, 1),
                blocked_v.view(-1, h_kv, dv)[begin:end].transpose(0, 1),
                h_q=h_q,
                h_kv=h_kv,
                is_causal=causal,
            )
            out[i] = O.transpose(0, 1)
            lse[i] = LSE
        return out, lse

    # out_flash, lse_flash = flash_mla()
    # out_torch, lse_torch = ref_mla()
    # cal_diff(out_flash, out_torch, "out")
    # cal_diff(lse_flash, lse_torch, "lse")

    # t = triton.testing.do_bench(flash_mla)
    t = tilelang.profiler.do_bench(flash_mla)
    FLOPS = s_q * b * seqlen_k * h_q * (d + dv) * 2
    bytes = (b*seqlen_k * h_kv * d + b * s_q * h_q * d + b * s_q * h_q * dv) * (
        torch.finfo(q.dtype).bits // 8
    )
    print(
        f"{t:.3f} ms, {FLOPS / 10 ** 9 / t:.0f} TFLOPS, {bytes / 10 ** 6 / t:.0f} GB/s"
    )

@torch.inference_mode()
def test_mod(mod, b, s_q, mean_sk, h_q, h_kv, d, dv, causal, varlen):
    import triton
    import tilelang
    from benchmark.bench_utils import print_debug
    # print(
    #     f"{b=}, {s_q=}, {mean_sk=}, {h_q=}, {h_kv=}, {d=}, {dv=}, {causal=}, {varlen=}"
    # )

    cache_seqlens = torch.full((b,), mean_sk, dtype=torch.int32, device="cuda")
    # if varlen:
    #     for i in range(b):
    #         cache_seqlens[i] = max(random.normalvariate(mean_sk, mean_sk / 2), s_q)
    # total_seqlens = cache_seqlens.sum().item()
    # mean_seqlens = cache_seqlens.float().mean().int().item()
    max_seqlen = cache_seqlens.max().item()
    max_seqlen_pad = triton.cdiv(max_seqlen, 256) * 256
    # print(f"{total_seqlens=}, {mean_seqlens=}, {max_seqlen=}")

    # tile_scheduler_metadata, num_splits = get_mla_metadata(
    #     cache_seqlens, s_q * h_q // h_kv, h_kv
    # )
    block_size = 64
    block_table = torch.arange(
        b * max_seqlen_pad // block_size, dtype=torch.int32, device="cuda"
    ).view(b, max_seqlen_pad // block_size)
    
    q = torch.ones(b, s_q, h_q, d, dtype=torch.float16, device="cuda")
    blocked_k = torch.randn(block_table.numel(), block_size, h_kv, d, dtype=torch.float16, device="cuda")
    for i in range(b):
        blocked_k.view(b, max_seqlen_pad, h_kv, d)[i, cache_seqlens[i].item():] = (
            float("nan")
        )
    blocked_v = blocked_k[..., :dv]


    def flash_mla():
        return mod(
            q, blocked_k
        )

    def ref_mla():
        out = torch.empty(b, s_q, h_q, dv, dtype=torch.float32, device="cuda")
        lse = torch.empty(b, h_q, s_q, dtype=torch.float32, device="cuda")
        for i in range(b):
            begin = i * max_seqlen_pad
            end = begin + cache_seqlens[i]
            O, LSE = scaled_dot_product_attention(
                q[i].transpose(0, 1),
                blocked_k.view(-1, h_kv, d)[begin:end].transpose(0, 1),
                blocked_v.view(-1, h_kv, dv)[begin:end].transpose(0, 1),
                h_q=h_q,
                h_kv=h_kv,
                is_causal=causal,
            )
            out[i] = O.transpose(0, 1)
            lse[i] = LSE
        return out, lse

    out_flash, lse_flash = flash_mla()
    out_torch, lse_torch = ref_mla()
    # cal_diff(out_flash, out_torch, "out")
    print_debug(out_flash.float(), out_torch)
    cal_diff(lse_flash, lse_torch, "lse")

    # t = triton.testing.do_bench(flash_mla)
    t = tilelang.profiler.do_bench(flash_mla)
    FLOPS = s_q * b * seqlen_k * h_q * (d + dv) * 2
    bytes = (b*seqlen_k * h_kv * d + b * s_q * h_q * d + b * s_q * h_q * dv) * (
        torch.finfo(q.dtype).bits // 8
    )
    print(
        f"{t:.3f} ms, {FLOPS / 10 ** 9 / t:.0f} TFLOPS, {bytes / 10 ** 6 / t:.0f} GB/s"
    )

if __name__ == "__main__":
    torch.cuda.manual_seed(0)
    B, H, G ,S, D, DV = 128, 128, 1, 4096, D, 512# 384
    # D = DV + D_pe = 512 + 64 = 576
    dtype = torch.bfloat16
    qkv_meta = (
        meta_tensor(B, H, 1, D, dtype=dtype),
        meta_tensor(B, G, S, D, dtype=dtype),
        meta_tensor(B, G, S, DV, dtype=dtype),
    )

    custom_fwd_inputs = CustomIO({
        
    })

    online = OnlineSoftmax()
    mod = AttentionEngine(
        qkv_meta,
        custom_fwd_inputs, score_mod=score_mod, mask_mod=None,
        online_func=online,
        kv_shared=True,
        backend="cute",
    )
    
    q = torch.randn(B, 1, H, D, dtype=dtype, device="cuda")
    KV = torch.randn(B*S//64,64, G, DV, dtype=dtype, device="cuda")
    k_pe = torch.randn(B*S//64,64, G, D-DV, dtype=dtype, device="cuda")
    KV = torch.concat([KV, k_pe], dim=-1).contiguous()
    
    o = mod(
        q,
        KV
    )
    from tilelang.profiler import do_bench
    latency = do_bench(
        lambda: mod(
            q,
            KV,
        ),
        # warmup=10,
        # rep=100,
    )
    print("latency: ", latency)
    flops = B * S * H * (D + DV) * 2
    print("flops/s: ", flops / latency)
    
    b = qkv_meta[0].shape[0]
    s_q = qkv_meta[0].shape[2]
    h_q = qkv_meta[0].shape[1]
    h_kv = qkv_meta[2].shape[1]
    seqlen_k = qkv_meta[2].shape[2]
    head_dim_v = qkv_meta[2].shape[3]
    
    # test_flash_mla(
    #     b,
    #     s_q,
    #     seqlen_k,
    #     h_q,
    #     h_kv,
    #     D,
    #     DV,
    #     causal=False,
    #     varlen=False
    # )
    test_mod(
       mod,
       b,
       s_q,
       seqlen_k,
       h_q,
       h_kv,
       D,
       DV,
       causal=False,
       varlen=False
   )
