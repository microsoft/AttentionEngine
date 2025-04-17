import torch
from core.transform.core import CustomIO, SymbolicArray, SymbolScalar, Var

from autotuner.decider import decider
from autotuner.arch import H100

import importlib.util
import tempfile
import os
import os.path as osp
import hashlib
from functools import partial
from typing import Optional, Callable, Union


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
    def __init__(self, qkv_meta, custom_fwd_inputs, score_mod, mask_mod,
                 online_func, mask_value="-inf", device=H100(), backend="tl", 
                 tune=False, tune_file="", 
                 tune_bwd=False, tune_file_bwd="",
                 infer_mask=False,
                 kernel_template=None):
        # tunner
        # need_engine_fuse, fuse_config = decider(qkv_meta, device)
        
        # if dynamic shape
        # TODO: 111

        # backend
        if backend == "tl":
            self._compile_tl(
                qkv_meta,
                custom_fwd_inputs,
                score_mod,
                mask_mod,
                online_func,
                mask_value, infer_mask=infer_mask, 
                tune=tune,
                tune_file=tune_file,
                tune_bwd=tune_bwd,
                tune_file_bwd=tune_file_bwd,
                kernel_template=kernel_template)

        elif backend == "cute":
            # must be same with cute_template.py
            OUTPUT_DIR = osp.join(
                osp.dirname(
                    osp.abspath(__file__)),
                "../core/template/cute_template_output")
            cutlass_dtype_map = {
                torch.float16: "cutlass::half_t",
                torch.bfloat16: "cutlass::bfloat16_t",
            }
            file_path = os.path.join(OUTPUT_DIR, "flash_attn_interface.py")
            lower_cute(score_mod,
                       mask_mod,
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
                causal=True if mask_mod is not None else False)

    def _select_lower_template(self, qkv_meta, custom_fwd_inputs, score_mod, mask_mod,
                    online_func, mask_value="-inf", tuned_config=None, infer_mask=False,
                    tune=False, tune_file="",
                    tune_bwd=False, tune_file_bwd="",
                    kernel_template=None):
        tl_dtype_map = {
            torch.float16: "float16",
            torch.bfloat16: "bfloat16",
        }
        q_seqlen = qkv_meta[0].shape[2]
        kv_len = qkv_meta[2].shape[2]
        head = qkv_meta[0].shape[1]
        head_kv = qkv_meta[2].shape[1]
        
        # mla decode
        if kernel_template == "mla_decode":
            from core.lower.lower_decode_mla import lower_tl as lower_tl_decode_mla
            tl_code = lower_tl_decode_mla(score_mod,
                                        mask_mod,
                                        online_func,
                                        custom_fwd_inputs,
                                        qkv_meta[0].shape[0], # B
                                        head, # headq
                                        head_kv, # H
                                        kv_len, # S
                                        qkv_meta[0].shape[3],
                                        qkv_meta[2].shape[3],
                                        tl_dtype_map[qkv_meta[0].dtype],
                                        mask_value,
                                        tuned_config)
            return tl_code, None
        
        # decode gqa
        if q_seqlen != kv_len and head > head_kv: # TODO: change condition
            assert (q_seqlen < kv_len)
            assert q_seqlen == 1
            infer_mask = True
            from core.lower.lower_decode_gqa import lower_tl as lower_tl_decode_gqa
            tl_code, block_mask = lower_tl_decode_gqa(score_mod,
                                      mask_mod,
                                      online_func,
                                      custom_fwd_inputs,
                                      qkv_meta[0].shape[0], # B
                                      head, # headq
                                        head_kv, # H
                                        kv_len, # S
                                      qkv_meta[0].shape[3],
                                      qkv_meta[2].shape[3],
                                      tl_dtype_map[qkv_meta[0].dtype],
                                      mask_value,
                                      tuned_config)
            return tl_code, block_mask
            
        # decode mha
        if q_seqlen != kv_len and head == head_kv:
            assert (q_seqlen < kv_len)
            from core.lower.lower_decode import lower_tl as lower_tl_decode
            tl_code = lower_tl_decode(score_mod,
                                      mask_mod,
                                      online_func,
                                      custom_fwd_inputs,
                                      qkv_meta[0].shape[3],
                                      qkv_meta[2].shape[3],
                                      tl_dtype_map[qkv_meta[0].dtype],
                                      mask_value,
                                      tuned_config)
            return tl_code, None
        
        # train/prefill mha forward & backward
        if q_seqlen == kv_len and head == head_kv:
            from core.lower.lower import lower_tl
            tl_code, block_mask = lower_tl(score_mod,
                                mask_mod,
                                online_func,
                                custom_fwd_inputs,
                                qkv_meta[0].shape[0], # B
                                qkv_meta[0].shape[1], # H
                                q_seqlen, # S
                                qkv_meta[0].shape[3],
                                qkv_meta[2].shape[3],
                                tl_dtype_map[qkv_meta[0].dtype],
                                mask_value,
                                tuned_config, infer_mask, 
                                tune=tune, tune_file=tune_file,
                                tune_bwd=tune_bwd, tune_file_bwd=tune_file_bwd)
            return tl_code, block_mask
        
        # gqa forward & backward
        if q_seqlen == kv_len and head > head_kv:
            from core.lower.lower_gqa import lower_tl
            tl_code, block_mask = lower_tl(score_mod,
                                mask_mod,
                                online_func,
                                custom_fwd_inputs,
                                qkv_meta[0].shape[0], # B
                                qkv_meta[0].shape[1], # H
                                q_seqlen, # S
                                qkv_meta[0].shape[3],
                                qkv_meta[2].shape[3],
                                tl_dtype_map[qkv_meta[0].dtype],
                                mask_value,
                                tuned_config, infer_mask, 
                                tune=tune, tune_file=tune_file,
                                tune_bwd=tune_bwd, tune_file_bwd=tune_file_bwd)
            return tl_code, block_mask
        
    def _compile_tl(self, qkv_meta, custom_fwd_inputs, score_mod, mask_mod,
                    online_func, mask_value="-inf", tuned_config=None, infer_mask=False,
                    tune=False, tune_file="",
                    tune_bwd=False, tune_file_bwd="",
                    kernel_template=None):
        tl_dtype_map = {
            torch.float16: "float16",
            torch.bfloat16: "bfloat16",
        }
        q_seqlen = qkv_meta[0].shape[2]
        kv_len = qkv_meta[2].shape[2]
        head = qkv_meta[0].shape[1]
        head_kv = qkv_meta[2].shape[1]
        block_mask = None
        tl_code, block_mask = self._select_lower_template(
            qkv_meta,
            custom_fwd_inputs,
            score_mod,
            mask_mod,
            online_func,
            mask_value, tuned_config=tuned_config, infer_mask=infer_mask,
            tune=tune,
            tune_file=tune_file,
            tune_bwd=tune_bwd,
            tune_file_bwd=tune_file_bwd,
            kernel_template=kernel_template
        )
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
        self.attention = tl_attn.attention
        if infer_mask:
            self.block_mask = block_mask
        else:
            self.block_mask = None

    def __call__(self, *args, **kargs):
        if self.block_mask is not None:
            o = self.attention(*args, self.block_mask, **kargs)
        else:
            o = self.attention(*args, **kargs)
        return o

