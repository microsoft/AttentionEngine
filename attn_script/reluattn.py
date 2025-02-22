from attn_engine import AttentionEngine
import torch
import math
from attn_engine import OnlineFunc
from core.core import CustomIO
from core.core import create_block_mask
from core.core import SymbolicArray, SymbolScalar, SymbolicTensor
from core.core import Var
from core.utils import meta_tensor



D = 64
scores_scale = 1/D**0.5
def score_mod(score, custom_fwd_inputs, b, h, q_idx, kv_idx):
    score = score * scores_scale
    score = score.max(0)
    return score


class OnlineIdentity(OnlineFunc):
    def __init__(self):
        online_rowscales = {
        }
        final_rowscales = {
        }
        external_fwd_inputs = CustomIO()
        super().__init__(online_rowscales, final_rowscales,
                    external_fwd_inputs)

    @staticmethod
    def online_fwd(scores, online_rowscales, b, h, q_idx):
        o_scale = SymbolScalar("o_scale", Var("1"))
        return scores, online_rowscales, o_scale
    
    @staticmethod
    def online_fwd_epilogue(o, online_rowscales, b, h, q_idx):
        return o, {}
    
    @staticmethod
    def forward(scores, final_rowscales, b, h, q_idx, kv_idx):
        return scores

    @staticmethod
    def backward(dp, scores, final_rowscales, doosum_rowscales, b, h, q_idx, kv_idx):
        return dp

custom_fwd_inputs = CustomIO({
})

def eval():
    import itertools
    BHSD = list(
        itertools.product(
            (32,64,),
            (6,),
            (512,1024, 2048),
            (64,),
            (64,)
        )
    )
    for B, H, S, D, DV in BHSD:
        print(f"B={B}, H={H}, S={S}, D={D}, DV={DV}")
        qkv_meta = (
            meta_tensor(B, H, S, D, dtype=torch.float16),
            meta_tensor(B, H, S, D, dtype=torch.float16),
            meta_tensor(B, H, S, D, dtype=torch.float16),
        )

        mod = AttentionEngine(
            qkv_meta,
            custom_fwd_inputs, score_mod=score_mod, block_mask=None,
            online_func=OnlineIdentity(),
            tune = True, tune_file = "reluattn_tune.json"
        )
        with open("reluattn_tl_code.py", "w") as f:
            f.write(mod.tl_code)
        from benchmark.bench_utils import do_bench_reluattn
        do_bench_reluattn(mod, B, H, S, D, D)
if __name__ == "__main__":
    B, H ,S, D = 64,6,2048,D
    qkv_meta = (
        meta_tensor(B, H, S, D, dtype=torch.float16),
        meta_tensor(B, H, S, D, dtype=torch.float16),
        meta_tensor(B, H, S, D, dtype=torch.float16),
    )

    mod = AttentionEngine(
        qkv_meta,
        custom_fwd_inputs, score_mod=score_mod, block_mask=None,
        online_func=OnlineIdentity(),
        tune = True, tune_file = "reluattn_tune.json"
    )
    from benchmark.bench_utils import do_bench_reluattn
    do_bench_reluattn(mod, B, H, S, D, D, requires_grad=True)
    # eval()

