import torch
from core.lower import lower_tl
from core.lower_decode import lower_tl as lower_tl_decode
from core.lower_cute import lower_cute
from core.core import CustomIO, SymbolicArray, SymbolScalar, Var

from autotuner.decider import decider
from autotuner.arch import H100

import importlib.util
import tempfile
import os
import os.path as osp
import hashlib
from functools import partial


class OnlineFunc:
    """
    __init__: define online_rowscales and final_rowscales
        online_rowscales: intermediate scale results for online algorithm
        final_rowscales: final scale results for online algorithm

    online_fwd: online algorithm for generate attention forward

    set_final_rowscales: set final rowscales at the end of attention forward, save it for backward

    forward: forward algorithm g(scores, scale) for backward recompute
    backward: backward algorithm
    """

    def __init__(self, online_rowscales: dict[str, SymbolScalar], final_rowscales: dict[str, SymbolScalar],
                 external_fwd_tensors: CustomIO):  # , external_bwd_tensors:CustomIO):
        # TODO: external_tensors
        """
        define&init online_rowscales and final_rowscales
        """
        self.online_rowscales = online_rowscales
        self.final_rowscales = final_rowscales
        self.vars = {
            "scores": SymbolicArray(),
            "o_scale": None,
        }
        self.external_fwd_tensors = external_fwd_tensors
        # self.external_bwd_tensors = external_bwd_tensors
        self.doosum_rowscales = SymbolicArray(
            "doosum", Var("doosum"), shape_idx=["block_M"])

    @staticmethod
    def online_fwd(scores: SymbolicArray, online_rowscales, b, h, q_idx):
        """
        compute scores, online_rowscale, o_scale
        input:
            scores: symbolic tensor, including method like getreduce()
            online_rowscales: the intermediate scale results of the previous round
        return:
            scores: symbolic tensor
            online_rowscales: save the updated intermediate results of the online algorithm
            o_scale:  for online rescale o

        """
        o_scale = SymbolScalar("o_scale", Var("o_scale"))
        return scores, online_rowscales, o_scale

    @staticmethod
    def online_fwd_epilogue(o, online_rowscales, b, h, q_idx):
        """
        compute o, final_rowscales at the end of online attention forward
        return:
            o: symbolic tensor
            final_rowscales: save the final scale results of the online algorithm, used for backward
        """
        final_rowscales = online_rowscales
        return o, final_rowscales

    @staticmethod
    def forward(
            scores, final_rowscales: dict[str, SymbolScalar], b, h, q_idx, kv_idx):
        """
        compute scores : scores = g(scores, scale),
            final_rowscales is saved during online forward
        return
        """
        return scores

    @staticmethod
    def backward(
            dp, scores, final_rowscales: dict[str, SymbolScalar], b, h, q_idx, kv_idx):
        """
        compute bwd scores: dscores = g_bwd(dp, scores)
        only support elementwise
        """
        dscores = dp
        return dscores


class AttentionEngine:
    def __init__(self, qkv_meta, custom_fwd_inputs, score_mod, block_mask,
                 online_func, mask_value="-inf", device=H100(), backend="tl", tune=False, tune_file="tuned_result.json"):
        # tunner
        need_engine_fuse, fuse_config = decider(qkv_meta, device)

        # backend
        if backend == "tl":
            self._compile_tl(
                qkv_meta,
                custom_fwd_inputs,
                score_mod,
                block_mask,
                online_func,
                mask_value)

            if tune:
                from autotuner.attnfwd_tunner_engine2 import AttnFwdTunner
                TUNE_SPACE = {
                    "block_M": [64, 128, 256],
                    "block_N": [32, 64, 128, 256],
                    "stages": [1, 2],  # ,3],
                    "num_threads": [128, 256],
                    "shared_fuse": [True, False],
                }
                B, H, S, DK = qkv_meta[0].shape
                DV = qkv_meta[2].shape[3]
                output_idx_list = [i for i in range(3 +
                                                    len(custom_fwd_inputs.input_tensors), 3 +
                                                    len(custom_fwd_inputs.input_tensors) +
                                                    1 +
                                                    len(online_func.final_rowscales))]
                tl_kernel = self.kernel

                st = AttnFwdTunner(DK, DV, **TUNE_SPACE)
                configs = st.generate_config()
                print(configs)
                # program = tl_kernel(B, H, S, DK, DV, *configs[0].values())
                # import tilelang as tl
                # mod, params = tl.lower(program)
                problem_keys = {
                    "B": B, "H": H, "N_CTX": S, "D_HEAD": DK, "D_HEADV": DV  # , "causal":True
                }
                tuned_config = st.tl_tune(
                    tl_kernel,
                    problem_keys,
                    configs,
                    output_idx_list,
                    file_path=tune_file)
                self._compile_tl(
                    qkv_meta,
                    custom_fwd_inputs,
                    score_mod,
                    block_mask,
                    online_func,
                    mask_value,
                    tuned_config)

        elif backend == "cute":
            # must be same with cute_template.py
            OUTPUT_DIR = osp.join(
                osp.dirname(
                    osp.abspath(__file__)),
                "../core/cute_template_output")
            cutlass_dtype_map = {
                torch.float16: "cutlass::half_t",
                torch.bfloat16: "cutlass::bfloat16_t",
            }
            file_path = os.path.join(OUTPUT_DIR, "flash_attn_interface.py")
            lower_cute(score_mod,
                       block_mask,
                       online_func,
                       custom_fwd_inputs,
                       qkv_meta[0].shape[3],
                       qkv_meta[2].shape[3],
                       cutlass_dtype_map[qkv_meta[0].dtype])
            spec = importlib.util.spec_from_file_location(
                "cute_attn", file_path)
            cute_attn = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cute_attn)
            # TODO: causal
            self.attention = partial(
                cute_attn.flash_attn_func,
                causal=True if block_mask is not None else False)

    def _compile_tl(self, qkv_meta, custom_fwd_inputs, score_mod, block_mask,
                    online_func, mask_value="-inf", tuned_config=None):
        tl_dtype_map = {
            torch.float16: "float16",
            torch.bfloat16: "bfloat16",
        }
        q_seqlen = qkv_meta[0].shape[2]
        kv_len = qkv_meta[2].shape[2]
        if q_seqlen != kv_len:
            assert (q_seqlen < kv_len)
            tl_code = lower_tl_decode(score_mod,
                                      block_mask,
                                      online_func,
                                      custom_fwd_inputs,
                                      qkv_meta[0].shape[3],
                                      qkv_meta[2].shape[3],
                                      tl_dtype_map[qkv_meta[0].dtype],
                                      mask_value,
                                      tuned_config)
        else:
            tl_code = lower_tl(score_mod,
                               block_mask,
                               online_func,
                               custom_fwd_inputs,
                               qkv_meta[0].shape[3],
                               qkv_meta[2].shape[3],
                               tl_dtype_map[qkv_meta[0].dtype],
                               mask_value,
                               tuned_config)
        self.tl_code = tl_code  
        # for debug
        # with open("generated_tl.py","w") as f:
        #      f.write(tl_code)
        code_hash = hashlib.md5(tl_code.encode()).hexdigest()
        cache_dir = os.path.join(os.path.dirname(__file__), "cache")
        file_path = os.path.join(cache_dir, f"{code_hash}.py")
        os.makedirs(cache_dir, exist_ok=True)
        if not os.path.exists(file_path):
            with open(file_path, "w") as f:
                f.write(tl_code)
                f.flush()
        # replace code
        # file_path = "/home/aiscuser/cfy/AttentionEngine/attn_script/generated_tl_code_attention.py"
        spec = importlib.util.spec_from_file_location("tl_attn", file_path)
        tl_attn = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tl_attn)
        self.kernel = tl_attn.kernel
        self.attention = tl_attn.attention

    def __call__(self, *args, **kargs):

        o = self.attention(*args, **kargs)
        return o


if __name__ == "__main__":
    online = OnlineFunc({}, {}, CustomIO(), CustomIO())
    scores, online_rowscales, o_scale = online.online_fwd(
        SymbolicArray(), online.online_rowscales, 1, 1, 1)
    o, final_scales = online.online_fwd_epilogue(
        SymbolScalar("o", Var("o")), online.online_rowscales, 1, 1, 1)
    scores2 = online.forward(
        SymbolicArray(),
        online.final_rowscales,
        1,
        1,
        1,
        1)
    dscores = online.backward(
        SymbolScalar(
            "dp",
            Var("dp")),
        SymbolScalar(
            "scores",
            Var("scores")),
        online.final_rowscales,
        online.external_bwd_tensors,
        1,
        1,
        1,
        1)
