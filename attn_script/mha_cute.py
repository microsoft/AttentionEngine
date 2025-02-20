from attn_engine import AttentionEngine
import torch
import math
from attn_engine import OnlineFunc
from core.core import CustomIO
from core.core import create_block_mask
from core.core import SymbolicArray, SymbolScalar, SymbolicTensor
from core.core import Var
from core.utils import meta_tensor

"""
Example of causal attention with online softmax
"""



# mask on attention score
def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

D = 64 # 128
softmax_scale = (1/D) ** 0.5
softmax_scale_log2e = softmax_scale * math.log2(math.e)
# elementwise on attention scores
def score_mod(score, custom_fwd_inputs, b, h, q_idx, kv_idx):
    # softmax_scale = custom_fwd_inputs.input_tensors["softmax_scale"]
    return score

class OnlineSoftmax(OnlineFunc):
    def __init__(self):
        """
        define online_rowscales and final_rowscales
        """
        online_rowscales = {
            "m": SymbolScalar("m", Var("-inf")), # -inf 
            "r": SymbolScalar("r", Var("0.0")),
        }
        final_rowscales = {
            "lse": SymbolScalar("lse", Var("0.0")), # not used in codegen
        }
        external_fwd_inputs = CustomIO()
        # external_bwd_inputs = CustomIO({
        #     "dppsum": (B,H,S),
        # })
        super().__init__(online_rowscales, final_rowscales,
                    external_fwd_inputs) # , external_bwd_inputs)
    

    @staticmethod
    def online_fwd(scores, online_rowscales, b, h, q_idx):

        m , r = online_rowscales["m"], online_rowscales["r"]
        m_new = m.max(scores.get_reduce("max"))
        scale_tmp = ((m - m_new)*softmax_scale_log2e).exp2()
        r = r * scale_tmp
        
        scores = ((scores - m_new)*softmax_scale_log2e).exp2()
        r = r + scores.get_reduce("sum")

        new_online_rowscales = {
            "m": m_new,
            "r": r,
        }
        o_scale = scale_tmp
        return scores, new_online_rowscales, o_scale

    @staticmethod
    def online_fwd_epilogue(o, online_rowscales, b, h, q_idx):
        o_new = o / online_rowscales["r"]
        lse = (online_rowscales["r"]).log() + online_rowscales["m"]*softmax_scale
        final_rowscales = {
            "lse": lse,
        }
        return o_new, final_rowscales

    @staticmethod
    def forward(scores, final_rowscales, b, h, q_idx, kv_idx):
        lse = final_rowscales["lse"]
        scores_new = (scores*softmax_scale-lse).exp()
        return scores_new
    
    @staticmethod
    def backward(dp, scores, final_rowscales, doosum_rowscales, b, h, q_idx, kv_idx):
        dppsum = doosum_rowscales # external_bwd_tensor.input_tensors["dppsum"]
        dscores = (dp - dppsum)*scores # TODO: bug if swap
        return dscores      

if __name__ == "__main__":
    B, H ,S, D,DV = 1,12,2048,D,64
    dtype = torch.bfloat16
    qkv_meta = (
        meta_tensor(B, H, S, D, dtype=dtype),
        meta_tensor(B, H, S, D, dtype=dtype),
        meta_tensor(B, H, S, DV, dtype=dtype),
    )

    custom_fwd_inputs = CustomIO({
        # "softmax_scale": (1,),
    })

    online = OnlineSoftmax()
    block_mask = create_block_mask(causal_mask, 1, 1, S, S, device="cuda")

    mod = AttentionEngine(
        qkv_meta,
        custom_fwd_inputs, score_mod=score_mod, block_mask=block_mask,
        online_func=online,
        backend="cute"
    )


    from benchmark.bench_utils import do_bench_attention
    do_bench_attention(mod, B, H, S, D, DV, dtype=dtype, require_grad=True)
