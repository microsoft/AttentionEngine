# AttentionEngine

AttentionEngine is a unified framework to customize attention, including transformer attention and linear attention. AttentionEngine provides users with pythonic interface to define customized attention **flexibly** and automatically generate device code with **high performance**. For example,  user can define softmax attention with only 80 lines of code and get the optimized fused kernel automatically.

# Tested Devices

AttentionEngine aims to support multiple backends, including NVIDIA GPUs and AMD GPUs. Currently, it has been specifically tested and validated on the following devices:
- NVIDIA H100
- AMD MI250 (TODO)

# Customized Attention Examples

Customized attention examples are under folder `attn_script`, including:
+ Transformer Attention
    - `attn_script/mha.py`: softmax attention
    - `attn_script/sigmoidattn.py`: sigmoid attention
    - `attn_script/reluattn.py`: relu attention
    - `attn_script/retention.py`: retnet attention
+ Linear Attention
    - `attn_script/mamba2_ngroup1.py`: mamba2
    - `attn_script/simple_gla.py`: gated retention
    - `attn_script/retnetion_linear.py`: retnet linear

# Benchmark Summary

AttentionEngine achieves exceptional performance across a variety of customized attention. Below are selected results showcasing its capabilities:
- softmax attention of LLAMA-3.1-8B shape on H100
- relu attention on H100
- mamba2 SSM on H100


# Installation
- install cuda==12.4 & pytorch, or use the docker image[Recommended]
```
pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel
```
- clone the repo and its submodule
```
git clone --recursive https://github.com/smallscientist1/AttentionEngine.git
```
- install **TileLang**: change directory `cd 3rd_parties/tilelang` and build TileLang from source according to this link (https://github.com/tile-ai/tilelang/blob/main/docs/get_started/Installation.md#method-2-install-from-source-using-the-bundled-tvm-submodule)
- export some environment variables
```
export PYTHONPATH="$(pwd)/attention_engine:$(pwd)/3rd_parties/tilelang:$PYTHONPATH"
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so
```

# Quick Start

In this section, you'll learn how to write and execute softmax attention using AttentionEngine.

```python
from attn_engine import AttentionEngine
import torch
from attn_engine import OnlineFunc
from core.core import CustomIO
from core.core import create_block_mask
from core.utils import meta_tensor

D = 128
softmax_scale = 1/D ** 0.5
# define elementwise modification on attention scores
def score_mod(score, custom_fwd_inputs, b, h, q_idx, kv_idx):
    return score * softmax_scale

# define custom inputs, here none
custom_fwd_inputs = CustomIO({
})

# define OnlineSoftmax as a subclass of OnlineFunc
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
        external_fwd_inputs = custom_fwd_inputs
        super().__init__(online_rowscales, final_rowscales,
                    external_fwd_inputs)
    

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

# mask on attention score
def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

if __name__ == "__main__":
    # define input shape
    B, H ,S, D, DV = 1,128,32768,D, 128
    dtype = torch.float16 # performance regression for bfloat16
    qkv_meta = (
        meta_tensor(B, H, S, D, dtype=dtype),
        meta_tensor(B, H, S, D, dtype=dtype),
        meta_tensor(B, H, S, DV, dtype=dtype),
    )
    block_mask = create_block_mask(causal_mask, 1, 1, S, S, device="cuda")

    # generate runtime attention op
    mod = AttentionEngine(
        qkv_meta,
        custom_fwd_inputs, score_mod=score_mod, block_mask=block_mask,
        online_func=OnlineSoftmax(),
        tune=False, tune_file="mha_tune.json"
    )

    # pytorch call
    q = torch.randn(B, S, H, D, dtype=dtype, device="cuda")
    k = torch.randn(B, S, H, D, dtype=dtype, device="cuda")
    v = torch.randn(B, S, H, DV, dtype=dtype, device="cuda")
    out = mod(q, k, v)
    out.backward(torch.ones_like(out))


    
```

# Roadmap
- [ ] Support backward on CuTe backend 
- [ ] Support decoding shape
- [ ] Support more sparse mask pattern
- [ ] Support AMD MI250


## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
