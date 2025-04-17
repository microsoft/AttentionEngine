# from ..attn_engine import OnlineFunc
from ..transform.core import SymbolScalar, SymbolicArray, CustomIO, is_causal_mask, is_less_causal_mask, create_block_mask
from ..transform.graph import Var, Const
from ..utils import IndentedCode
from ..codegen.tl_gen import generate_tl_from_dag
from ..template.attn_template import TlAttnTemplate
from ..template.blockattn_template import TlBlockAttnTemplate
from dataclasses import dataclass, field, InitVar

from ..codegen.common import *
from copy import copy, deepcopy
from sympy import symbols
from typing import Callable
import sympy as sp

import logging

import torch.fx as fx
import os.path as osp
THIS_FILE_PATH = osp.dirname(osp.abspath(__file__))
TEMPLATE_PATH = osp.join(
    THIS_FILE_PATH,
    "../template/tl_template/attn/attn_gqa_tl.py")
accum_type = "float"

shape_idx_map_sp = {
    "batch": sp.simplify("bz"),
    "heads": sp.simplify("by"),
    "seq_len": sp.simplify("bx*block_M"),
    "seq_len_kv": sp.simplify("k*block_N"),
    "1": sp.simplify("0")
    # others: ":" -> ":"
}
shape_idx_onchip_map = {
    "batch": "",
    "heads": "",
    "seq_len": "block_M",
    "seq_len_kv": "block_N",
    "1": ""
}
shape_idx_onchip_step_map_sp = {
    "batch": sp.simplify("0"),
    "heads": sp.simplify("0"),
    "seq_len": sp.simplify("block_M"),
    "seq_len_kv": sp.simplify("block_N"),
    "1": sp.simplify("0")
}

shape_idx_onchip_dim_map = [
    "seq_len", "seq_len_kv"
]

shape_idx_map_bwd = {
    "batch": "bz",
    "heads": "by",
    "seq_len": "k*block_N:(k+1)*block_N",
    "seq_len_kv": "bx*block_M:(bx+1)*block_M",
    "1": "0"
}
shape_idx_onchip_map_bwd = {
    "batch": "",
    "heads": "",
    "seq_len": "block_N",
    "seq_len_kv": "block_M",
    "1": ""
}

RECURRENT_DIM = "block_N"

@dataclass
class CopyMap:
    src: SymbolScalar
    dst: SymbolScalar
    idx_list: List[sp.Symbol]
    idx_dim_map: List[int]
    
@dataclass
class KernelOptionsBase:
    global_tensors_input: Dict[str, SymbolScalar] = field(default_factory=dict)
    global_tensors_output: Dict[str, SymbolScalar] = field(default_factory=dict)
    shared_tensors: Dict[str, SymbolScalar] = field(default_factory=dict)
    fragment_tensors: Dict[str, SymbolScalar] = field(default_factory=dict)
    copy_maps: List[CopyMap] = field(default_factory=list)
    
    def add_output_tensor(self, tensor_name: str, tile_shape, is_sharedmem: bool, global_shape, dtype:str, global_name=None, global_idx: List[sp.Symbol]=None, global_dim_map: List[int]=None):
        if global_name is None:
            global_name = f"g_{tensor_name}"
        new_tensor = SymbolScalar(tensor_name, Var(tensor_name), shape_idx=tile_shape, dtype=dtype)
        if is_sharedmem:
            if tensor_name in self.shared_tensors.keys():
                logging.warning(f"Tensor {tensor_name} already exists in shared memory")
            self.shared_tensors[tensor_name] = new_tensor
        else:
            if tensor_name in self.fragment_tensors.keys():
                logging.warning(f"Tensor {tensor_name} already exists in fragment memory")
            self.fragment_tensors[tensor_name] = new_tensor
        if global_idx is not None:
            self.global_tensors_output[global_name] = SymbolScalar(global_name, Var(global_name), shape_idx=global_shape, dtype=dtype)
            self.copy_maps.append(
                CopyMap(
                    new_tensor,
                    self.global_tensors_output[global_name],
                    global_idx,
                    global_dim_map,
                )
            )
        
    def add_input_tensor(self, tensor_name: str, tile_shape, is_sharedmem: bool, global_shape, dtype:str, global_idx: List[sp.Symbol]=None, global_dim_map: List[int]=None):
        new_tensor = SymbolScalar(tensor_name, Var(tensor_name), shape_idx=tile_shape, dtype=dtype)
        if is_sharedmem:
            if tensor_name in self.shared_tensors.keys():
                logging.warning(f"Tensor {tensor_name} already exists in shared memory")
            self.shared_tensors[tensor_name] = new_tensor
        else:
            if tensor_name in self.fragment_tensors.keys():
                logging.warning(f"Tensor {tensor_name} already exists in fragment memory")
            self.fragment_tensors[tensor_name] = new_tensor
        if global_idx is not None:
            self.global_tensors_input[f"g_{tensor_name}"] = SymbolScalar(f"g_{tensor_name}", Var(f"g_{tensor_name}"), shape_idx=global_shape, dtype=dtype)
            self.copy_maps.append(
                CopyMap(
                    self.global_tensors_input[f"g_{tensor_name}"],
                    new_tensor,
                    global_idx,
                    global_dim_map,
                )
            )
            
    def add_intermediate_tensor(self, tensor_name: str, shape, is_sharedmem: bool, dtype:str):
        tensor = SymbolScalar(tensor_name, Var(tensor_name), shape_idx=shape, dtype=dtype)
        if is_sharedmem:
            if tensor_name in self.shared_tensors.keys():
                logging.warning(f"Tensor {tensor_name} already exists in shared memory")
            self.shared_tensors[tensor.name] = tensor
        else:
            if tensor_name in self.fragment_tensors.keys():
                logging.warning(f"Tensor {tensor_name} already exists in fragment memory")
            self.fragment_tensors[tensor.name] = tensor
    
# AttnFwd Code string Shape&type, this class is not directly used to tl codegen
@dataclass
class AttnFwdKernelOption(KernelOptionsBase):
    tile_M: sp.Symbol = field(default_factory=sp.Symbol)
    tile_N: sp.Symbol = field(default_factory=sp.Symbol)
    accum_type: str = "float"
    dim: sp.Symbol = field(default_factory=sp.Symbol)
    dimv: sp.Symbol = field(default_factory=sp.Symbol)

# AttnFwd tuned config number
@dataclass
class TunnerOutput:
    TUNE: str = "False"
    TUNE_FILE: str = ""
    block_M: str = "128"
    block_N: str = "128"
    stages: str = "2"
    thread_num: str = "256"
    shared_fuse: str = "False"
    
@dataclass
class AttnBwdKernelOption(KernelOptionsBase):
    tile_M: sp.Symbol = field(default_factory=sp.Symbol)
    tile_N: sp.Symbol = field(default_factory=sp.Symbol)
    accum_type: str = "float"
    dim: sp.Symbol = field(default_factory=sp.Symbol)
    dimv: sp.Symbol = field(default_factory=sp.Symbol)
        
@dataclass
class TunnerOutputBwd:
    TUNE_BWD: str = "False"
    TUNE_FILE_BWD: str = ""
    block_M_bwd: str = "128"
    block_N_bwd: str = "64"
    thread_num_bwd: str = "256"
    
# Generated File: Code string
@dataclass
class lowerOutput:
    swizzle_shared: str = ""
    tl_dtype: str = "float16"
    is_inf_mask: str = "True"
    is_casual: str = "False"
    
    # problem shape
    BATCH: str = "1"
    HEADS: str = "1"
    SEQ_LEN: str = "1"
    DIM: str = "1"
    DIMV: str = "1"
    
    # mask_mod name&code
    q_idx: str = "q_idx"
    kv_idx: str = "kv_idx"
    batch_idx: str = "batch_idx"
    head_idx: str = "head_idx"
    mask_output: str = "True"
    mask_mod_code: str = ""
    is_mask_mod_code: str = "False"
    
    # score_mod name&code
    scores: str = "scores"
    
    # online_func name&code
    scores_online: str = "scores"
    acc_o: str = "acc_o"
    qkT: str = "qkT"
    dsT: str = "dsT"
    doosum_shared: str = "doosum_shared"



class lowerOnlineFuncOutput:

    def __init__(self, online_rowscales_initvalue, online_func_def, call_online_func, online_func_epilogue,
                 online_rowscales_update,
                 isused_doosum, final_rowscales_length, final_rowscales_load, online_func_fwd, custom_bwd_inputs_load, custom_bwd_body,
                 final_rowscales_shared_init,
                 custom_bwd_inputs, custom_bwd_inputs_init,
                 o_scale_varname):
        self.online_rowscales_initvalue = str(online_rowscales_initvalue)
        self.online_func_def = str(online_func_def)
        self.call_online_func = call_online_func
        self.online_func_epilogue = online_func_epilogue
        self.online_rowscales_update = str(online_rowscales_update)

        self.isused_doosum = isused_doosum
        self.final_rowscales_length = final_rowscales_length
        self.final_rowscales_load = final_rowscales_load
        self.online_func_fwd = online_func_fwd
        self.custom_bwd_inputs_load = custom_bwd_inputs_load
        self.custom_bwd_body = custom_bwd_body
        self.final_rowscales_shared_init = final_rowscales_shared_init
        self.custom_bwd_inputs = custom_bwd_inputs
        self.custom_bwd_inputs_init = custom_bwd_inputs_init
        self.o_scale_varname = o_scale_varname


@dataclass
class lowerScoreModOutput:
    score_mod_func_def: str
    call_score_mod: str
    score_mod_backward: str
    score_mod_bwd_inputs_list: str
    score_mod_bwd_inputs: str
    score_mod_inputs_bwd_list: str
    score_mod_fwd_inputs: str
    score_mod_fwd_body: str
    score_mod_output_var: str
    score_mod_bwd_inputs_declare: str
    score_mod_bwd_inputs_declare_shared: str
    
@dataclass
class customInputOutput:
    custom_fwd_inputs_load_shared: str = ""
    custom_fwd_inputs_load_s2r: str = ""
    custom_fwd_inputs_load_shared_bwd: str = ""

@dataclass
class lowerKernelBaseOutput:
    kernel_name: str
    input_args: str = ""
    output_args: str = ""
    alloc: str = ""
    output_args_copy_epilogue: str = ""
    input_args_copy_prologue: str = ""

def lower_kernel(kernel_options: KernelOptionsBase, kernel_template:lowerKernelBaseOutput):
    # generate input args
    input_args_code = IndentedCode()
    for tensor in kernel_options.global_tensors_input.values():
        input_args_code.add_line(arg_def(tensor))
    kernel_template.input_args = str(input_args_code)
    
    # generate output args
    output_args_code = IndentedCode()
    for tensor in kernel_options.global_tensors_output.values():
        output_args_code.add_line(arg_def(tensor))
    kernel_template.output_args = str(output_args_code)
    
    # generate alloc
    alloc_code = IndentedCode()
    for tensor in kernel_options.shared_tensors.values():
        alloc_code.add_line(alloc_shared_op(tensor))
    for tensor in kernel_options.fragment_tensors.values():
        alloc_code.add_line(alloc_fragment_op(tensor))
    kernel_template.alloc = str(alloc_code)
    
    # generate output args copy epilogue(reg/shared->global)
    output_args_copy_epilogue_code = IndentedCode()
    for copy_map in kernel_options.copy_maps:
        if copy_map.dst.name in kernel_options.global_tensors_output.keys():
            
            output_args_copy_epilogue_code.add_line(
                store_op(copy_map.src, copy_map.dst, copy_map.idx_dim_map, dst_dim_list=list(range(len(copy_map.idx_list))), dst_idx_list=copy_map.idx_list)
            )
    
    kernel_template.output_args_copy_epilogue = str(output_args_copy_epilogue_code)
    
    # generate input args copy prologue(global->reg/shared)
    input_args_copy_prologue_code = IndentedCode()
    for copy_map in kernel_options.copy_maps:
        if copy_map.src.name in kernel_options.global_tensors_input.keys():
            input_args_copy_prologue_code.add_line(
                load_op(copy_map.src, copy_map.dst, copy_map.idx_dim_map, src_dim_list=list(range(len(copy_map.idx_list))), src_idx_list=copy_map.idx_list)
            )
            
    kernel_template.input_args_copy_prologue = str(input_args_copy_prologue_code)


def lower_online_func(online_func, lower_output: lowerOutput,
                      kernel_options: AttnFwdKernelOption=None,
                      bwd_kernel_options: AttnBwdKernelOption=None):  
    online_fwd = online_func.online_fwd
    # 1. init input vars
    scores = SymbolicArray(
        lower_output.scores_online,
        Var(lower_output.scores_online),
        shape_idx=[
            kernel_options.tile_M,
            kernel_options.tile_N])
    online_rowscales = online_func.online_rowscales
    # TODO: support online_rowscales b h q_idx kv_idx
    b = SymbolScalar("b", Var("b"))
    h = SymbolScalar("h", Var("h"))
    q_idx = SymbolScalar("q_idx", Var("q_idx"))
    
    acco = SymbolicArray(lower_output.acc_o, Var(lower_output.acc_o), shape_idx=[kernel_options.tile_M, kernel_options.dimv])
    
    online_fwd_func_name = "online_func"

    # online&epilogue
    input_vars = {}

    # 2. fill op for online_rowscales
    online_rowscales_initvalue = IndentedCode()
    for k, v in online_rowscales.items():  # v.code is Var
        tl_init_value = v.code.name
        online_rowscales_initvalue.add_line(fill_op(v, tl_init_value))

    # 3. online_fwd func op def&call
    scores_new, new_online_rowscales, o_scalevar = online_fwd(
        scores, online_rowscales, b, h, q_idx)
    for k, v in new_online_rowscales.items():
        new_online_rowscales[k].set_allow_reuse(False)
    o_scalevar.set_allow_reuse(False)
    tl_code, input_vars_online = generate_tl_from_dag(
        list(new_online_rowscales.values()) + [scores_new, o_scalevar])
    online_func_def = func_block(
        online_fwd_func_name, input_vars_online.values(), tl_code
    )
    call_online_func = call_op(online_fwd_func_name, input_vars_online.values())
    input_vars.update(input_vars_online)
    
    # 4. o_scale update
    o_scale_varname = o_scalevar.varname

    # 4. online_rowscales update
    online_rowscales_update = IndentedCode()
    for k, v in new_online_rowscales.items():
        if v.varname == k:
            continue
        online_rowscales_update.add_line(
            copy_op(v, online_rowscales[k])
        )

    # 5. final_rowscales block
    for k, v in online_rowscales.items():
        online_rowscales[k].clear_codegen()
    acco_new, new_final_rowscales\
        = online_func.online_fwd_epilogue(acco, online_rowscales, b, h, q_idx)
    tl_code, input_vars_final = generate_tl_from_dag(
        [acco_new] + list(new_final_rowscales.values()))
    online_func_epilogue = str(tl_code)
    input_vars.update(input_vars_final)
    
    # 6. add kernel output tensor & intermediate tensor
    for k, v in new_final_rowscales.items():
        kernel_options.add_output_tensor(
            v.varname, v.shape, False, 
            ["batch", "heads", "seq_len"], v.dtype,
            f"g_{k}",
            [sp.simplify(ii) for ii in ["bz", "by", "bx * block_M"]],
            [2,]
        )
            
    for _, input_var in input_vars.items():
        if input_var.varname == scores.varname:
            continue
        kernel_options.add_intermediate_tensor(
            input_var.varname, input_var.shape_idx, False, input_var.dtype
        )


    # 7. bwd: TODO
    isused_doosum = False
    online_func_fwd = ""
    custom_bwd_inputs = ""
    custom_bwd_inputs_load = ""
    custom_bwd_inputs_init = ""
    custom_bwd_body = ""
    final_rowscales_load = ""
    final_rowscales_bwd = {}
    final_rowscales_length = 0
    final_rowscales_shared_init = ""
    if bwd_kernel_options is not None:
        qkT = SymbolScalar(lower_output.qkT, Var(lower_output.qkT), shape_idx=[str(bwd_kernel_options.tile_M), str(bwd_kernel_options.tile_N)])
        kv_idx = SymbolScalar("kv_idx", Var("kv_idx"))
        dsT = SymbolScalar(lower_output.dsT, Var(lower_output.dsT), shape_idx=[str(bwd_kernel_options.tile_M), str(bwd_kernel_options.tile_N)])
        doosum_shared = SymbolScalar(lower_output.doosum_shared, Var(lower_output.doosum_shared), shape_idx=["1", str(bwd_kernel_options.tile_N)])
            
        
        for k, v in online_func.final_rowscales.items():
            final_rowscales_bwd[k] = SymbolScalar(
                f"{k}_shared", Var(f"{k}"), shape_idx=[
                    "1", "block_N"])
        scores_2 = online_func.forward(
            qkT,
            final_rowscales_bwd,
            b,
            h,
            q_idx,
            kv_idx)

        tl_code, input_vars_fwd = generate_tl_from_dag([scores_2])
        online_func_fwd = str(tl_code)

        qkT.clear_codegen()
        dscores = online_func.backward(
            dsT, qkT, final_rowscales_bwd, doosum_shared, b, h, q_idx, kv_idx)

        tl_code, input_vars_bwd = generate_tl_from_dag([dscores])
        custom_bwd_body = str(tl_code)

        if lower_output.doosum_shared in input_vars_bwd:
            isused_doosum = True

        custom_bwd_inputs = f"g_doosum: T.Buffer([batch, heads, seq_len], accum_dtype), \n" if isused_doosum else ""
        for k, v in final_rowscales_bwd.items():
            final_rowscales_shared_init += f"{v.varname} = T.alloc_shared([{', '.join(v.shape_idx)}], accum_dtype, scope='shared')\n"
        custom_bwd_inputs_init = "doosum_shared = T.alloc_shared([1, block_N], accum_dtype, scope='shared')" if isused_doosum else ""

        for k, v in final_rowscales_bwd.items():
            final_rowscales_load += f"T.copy(g_{k}[bz, bx, k * block_N : (k + 1) * block_N], {v.varname})\n"
        custom_bwd_inputs_load = "T.copy(g_doosum[bz, bx, k * block_N : (k + 1) * block_N], doosum_shared)" if isused_doosum else ""
        final_rowscales_length = len(final_rowscales_bwd)

    return lowerOnlineFuncOutput(
        online_rowscales_initvalue=online_rowscales_initvalue,
        online_func_def=online_func_def,
        call_online_func=call_online_func,
        online_func_epilogue=online_func_epilogue,
        online_rowscales_update=online_rowscales_update,

        isused_doosum=isused_doosum,
        final_rowscales_length=final_rowscales_length,
        final_rowscales_load=final_rowscales_load,
        online_func_fwd=online_func_fwd,
        custom_bwd_inputs_load=custom_bwd_inputs_load,
        custom_bwd_body=custom_bwd_body,
        final_rowscales_shared_init=final_rowscales_shared_init,
        custom_bwd_inputs=custom_bwd_inputs,
        custom_bwd_inputs_init=custom_bwd_inputs_init,
        o_scale_varname=o_scale_varname
    )


def lower_score_mod(score_mod, custom_fwd_inputs, lower_output: lowerOutput, kernel_options: AttnFwdKernelOption, bwd_kernel_options: AttnBwdKernelOption):
    # 1. init input vars
    scores = SymbolScalar(
        lower_output.scores,
        Var(lower_output.scores),
        shape_idx=[
            str(kernel_options.tile_M),
            str(kernel_options.tile_N)])
    # TODO: support scoremod b h q_idx kv_idx
    b = SymbolScalar("b", Var("b"))
    h = SymbolScalar("h", Var("h"))
    q_idx = SymbolScalar("q_idx", Var("q_idx"))
    kv_idx = SymbolScalar("kv_idx", Var("kv_idx"))
    
    func_name = "score_mod"

    # 2. score_mod func op def&call
    scores_new = score_mod(scores, custom_fwd_inputs, b, h, q_idx, kv_idx)
    tl_code, input_vars = generate_tl_from_dag([scores_new])
    score_mod_func_def = func_block(func_name, input_vars.values(), tl_code)
    call_score_mod = call_op(func_name, input_vars.values())
    
    # 3. score_mod bwd TODO
    score_mod_backward = ""
    score_mod_inputs_bwd_list = ""
    score_mod_fwd_body = ""
    score_mod_output_var = ""
    score_mod_bwd_inputs_list = ""
    score_mod_bwd_inputs = IndentedCode()
    score_mod_fwd_inputs = IndentedCode()
    score_mod_bwd_inputs_declare = IndentedCode()
    score_mod_bwd_inputs_declare_shared = IndentedCode()
    if bwd_kernel_options is not None:

        # backward, block_M : k, block_N : q
        qkT = SymbolScalar(lower_output.qkT, Var(lower_output.qkT), shape_idx=[bwd_kernel_options.tile_M, bwd_kernel_options.tile_N])
        dsT = SymbolScalar(lower_output.dsT, Var(lower_output.dsT), shape_idx=[bwd_kernel_options.tile_M, bwd_kernel_options.tile_N])
        
        scores_new = score_mod(qkT, custom_fwd_inputs, b, h, q_idx, kv_idx)
        scores_new.backward(
            dsT)
        tl_code, input_vars_fwd = generate_tl_from_dag([scores_new])
        score_mod_fwd_body = str(tl_code)
        score_mod_output_var = scores_new.varname
        for varname, input_var in input_vars_fwd.items():
            score_mod_fwd_inputs.add_line(arg_def(input_var))
        score_mod_inputs_bwd_list = ", ".join(
            [varname for varname, input_var in input_vars_fwd.items()])
        tl_code, input_vars = generate_tl_from_dag([qkT.grad])
        score_mod_backward = str(tl_code)
        # add forward input_vars
        for k, v in input_vars_fwd.items():
            input_vars[k] = v
        score_mod_bwd_inputs_list = ", ".join(
            [input_var.varname for varname, input_var in input_vars.items()])
        for varname, input_var in input_vars.items():
            score_mod_bwd_inputs.add_line(arg_def(input_var))

        # TODO: dtype
        dtype = "accum_dtype"
        for varname, input_var in input_vars.items():
            if input_var.shape_idx == ["block_N", "block_M"]:
                dtype = "dtype"
                input_var.dtype = "dtype"
                score_mod_bwd_inputs_declare_shared.add_line(alloc_shared_op(input_var))
            else:
                input_var.dtype = dtype
                score_mod_bwd_inputs_declare.add_line(alloc_fragment_op(input_var))
                

    
    return lowerScoreModOutput(
        score_mod_func_def=str(score_mod_func_def),
        call_score_mod=str(call_score_mod),
        score_mod_backward=str(score_mod_backward),
        score_mod_bwd_inputs_list=str(score_mod_bwd_inputs_list),
        score_mod_bwd_inputs=str(score_mod_bwd_inputs),
        score_mod_inputs_bwd_list=str(score_mod_inputs_bwd_list),
        score_mod_fwd_inputs=str(score_mod_fwd_inputs),
        score_mod_fwd_body=str(score_mod_fwd_body),
        score_mod_output_var=str(score_mod_output_var),
        score_mod_bwd_inputs_declare=str(score_mod_bwd_inputs_declare),
        score_mod_bwd_inputs_declare_shared=str(score_mod_bwd_inputs_declare_shared)
    )

def lower_custom_inputs(custom_fwd_inputs, lower_output: lowerOutput, kernel_options: KernelOptionsBase):
    # deal with custom inputs tensors
    custom_fwd_inputs_load_shared_bwd = ""
    
    custom_fwd_inputs_load_shared = ""
    custom_fwd_inputs_load_s2r = ""
    for k, v in custom_fwd_inputs.input_tensors.items():
        # modify shape
        shape_idx_copy_sp = [(shape_idx_map_sp[shape] if shape in shape_idx_map_sp.keys(
        ) else sp.simplify("0")) for shape in v.shape_idx]
        shape_idx_block = [(shape_idx_onchip_map[shape] if shape in shape_idx_onchip_map.keys(
        ) else shape) for shape in v.shape_idx]
        # remove "" in list
        # TODO:bug[block_N] -> [1,block_N]
        shape_idx_block = [shape for shape in shape_idx_block if shape != ""]
        if shape_idx_block == []:
            shape_idx_block = ["1"]
        shape_idx_block_step_sp = [(shape_idx_onchip_step_map_sp[shape] if shape in shape_idx_onchip_step_map_sp.keys(
        ) else sp.simplify(shape)) for shape in v.shape_idx]
        shape_idx_dim_map = [idx for idx, shape in enumerate(v.shape_idx) if shape in shape_idx_onchip_dim_map]
        custom_input_dtype = "accum_dtype"
        
        kernel_options.fragment_tensors[k] = (SymbolScalar(k, Var(k), shape_idx=shape_idx_block, dtype=custom_input_dtype))
        kernel_options.global_tensors_input[f"g_{k}"] = (SymbolScalar(f"g_{k}", Var(f"g_{k}"), shape_idx=v.shape_idx, dtype=custom_input_dtype))
        
        # tl copy bug when "1"
        if not (RECURRENT_DIM in shape_idx_block):
            kernel_options.copy_maps.append(
                CopyMap(kernel_options.global_tensors_input[f"g_{k}"], kernel_options.fragment_tensors[k], shape_idx_copy_sp, shape_idx_dim_map)
            )
        elif len(shape_idx_block) > 1 and shape_idx_block[0] != "1":
            custom_input_dtype = "dtype"
            kernel_options.shared_tensors[f"{k}_shared"] = (SymbolScalar(f"{k}_shared", Var(f"{k}_shared"), shape_idx=shape_idx_block, dtype=custom_input_dtype))
            custom_fwd_inputs_load_shared += str(
                load_op(kernel_options.global_tensors_input[f"g_{k}"], kernel_options.shared_tensors[f"{k}_shared"], shape_idx_dim_map, src_dim_list=list(range(len(shape_idx_copy_sp))), src_idx_list=shape_idx_copy_sp) + "\n"
            )
            custom_fwd_inputs_load_s2r += copy_op(kernel_options.shared_tensors[f"{k}_shared"], kernel_options.fragment_tensors[k]) + "\n"
            lower_output.swizzle_shared += f"{k}_shared: tl.layout.make_swizzled_layout({k}_shared), \n"
        else:
            # custom_fwd_inputs_load_shared += f"T.copy(g_{k}[{', '.join(shape_idx_copy)}], {k})\n"
            custom_fwd_inputs_load_shared += str(
                load_op(kernel_options.global_tensors_input[f"g_{k}"], kernel_options.fragment_tensors[k], shape_idx_dim_map, src_dim_list=list(range(len(shape_idx_copy_sp))), src_idx_list=shape_idx_copy_sp) + "\n"
            )
        # TODO: dtype of custom_fwd_inputs
        
        kernel_options.fragment_tensors[k].dtype = custom_input_dtype
        kernel_options.fragment_tensors[k].shape_idx = shape_idx_block
        kernel_options.global_tensors_input[f"g_{k}"].dtype = custom_input_dtype
        custom_fwd_inputs.input_tensors[k].shape_idx = shape_idx_block
        
    return customInputOutput(
        custom_fwd_inputs_load_shared=custom_fwd_inputs_load_shared,
        custom_fwd_inputs_load_s2r=custom_fwd_inputs_load_s2r,
        custom_fwd_inputs_load_shared_bwd=custom_fwd_inputs_load_shared_bwd
    )


def lower_tl(score_mod, block_mask, online_func,
             custom_fwd_inputs,
             Batch, head, seqlen,
             dimqk, dimv, tl_dtype, mask_value, tuned_config=None, infer_mask=False,
             tune=False, tune_file="",
             tune_bwd=False, tune_file_bwd=""):

    # convert 0 to symbolic
    Batch = f"T.symbolic('{Batch}')" if isinstance(Batch, str) else Batch
    head = f"T.symbolic('{head}')" if isinstance(head, str) else head
    seqlen = f"T.symbolic('{seqlen}')" if isinstance(seqlen, str) else seqlen
    lower_output = lowerOutput(BATCH=str(Batch),
                              HEADS=str(head),
                              SEQ_LEN=str(seqlen),
                              DIM=str(dimqk),
                              DIMV=str(dimv))
    lower_output.tl_dtype = tl_dtype
    # TODO: mask_value: 0 or -inf
    lower_output.is_inf_mask = "True" if block_mask is not None and mask_value == "-inf" else "False"

    # ------------ATTN FWD----------------
    # 1. kernel performance configs
    # tune
    if tuned_config is None:
        tune_output = TunnerOutput(TUNE=str(tune), TUNE_FILE=str(tune_file))
    else:
        tune_output = TunnerOutput(**tuned_config)
    # Fwd config
    # TODO: remove this special check into autotuner
    if dimv > 256:
        tune_output.block_M = "64"
        tune_output.block_N = "64"
        tune_output.stages = "1"
        tune_output.shared_fuse = "True"
    if tune_output.shared_fuse == "True":
        lower_output.scores_online = "scores_1"

    # 2. kernel config options 
    kernel_options = AttnFwdKernelOption(tile_M=sp.simplify("block_M"), tile_N=sp.simplify("block_N"), 
                                         dim=sp.simplify("dim"), dimv=sp.simplify("dimv"))
    kernel_code_template = lowerKernelBaseOutput("kernel")
    
    
        
    # ------------ATTN BWD----------------
    # Bwd config(TODO: autotuner bwd)
    tune_output_bwd = TunnerOutputBwd(TUNE_BWD=str(tune_bwd), TUNE_FILE_BWD=str(tune_file_bwd))
    if max(dimqk, dimv) <= 64:
        tune_output_bwd.block_M_bwd = "128"
        tune_output_bwd.block_N_bwd = "128"
        tune_output_bwd.thread_num_bwd = "256"
    elif max(dimqk, dimv) <= 128:
        tune_output_bwd.block_M_bwd = "128"
        tune_output_bwd.block_N_bwd = "64"
        tune_output_bwd.thread_num_bwd = "256"
        
    bwd_kernel_options = AttnBwdKernelOption(tile_M=sp.simplify("block_M"), tile_N=sp.simplify("block_N"),
                                             dim=sp.simplify("dim"), dimv=sp.simplify("dimv"))
    
    # 3.kernel template specific lower
    # fwd&bwd

    lower_custom_inputs_output = lower_custom_inputs(
        custom_fwd_inputs, lower_output, kernel_options)
    
    lower_score_mod_output = lower_score_mod(
        score_mod, custom_fwd_inputs, lower_output, kernel_options, bwd_kernel_options)
    
    lower_online_func_output = lower_online_func(
        online_func, lower_output, kernel_options, bwd_kernel_options)
    output_idx_list = [i for i in range(3 +
                                        len(custom_fwd_inputs.input_tensors), 3 +
                                        len(custom_fwd_inputs.input_tensors) +
                                        1 +
                                        len(online_func.final_rowscales))]

    # TODO: custom_fwd_inputs grad
    bwd_output_idx_list = [i for i in range(4 +
                                            len(custom_fwd_inputs.input_tensors) +
                                            len(online_func.final_rowscales) +
                                            int(lower_online_func_output.isused_doosum), 4 +
                                            len(custom_fwd_inputs.input_tensors) +
                                            len(online_func.final_rowscales) +
                                            int(lower_online_func_output.isused_doosum) +
                                            3)]
    
    # 4. general kernel lower
    lower_kernel(kernel_options, kernel_code_template)
    
    # 5. mask mod
    if block_mask is not None:
        mask_graph = fx.symbolic_trace(block_mask)
        # TODO: check input and output
        node_list = [node for node in mask_graph.graph.nodes]
        lower_output.batch_idx = node_list[0].name
        lower_output.head_idx = node_list[1].name
        lower_output.q_idx = node_list[2].name
        lower_output.kv_idx = node_list[3].name
        lower_output.mask_output = node_list[-1].args[0].name
        lower_output.mask_mod_code = str(tl_codegen_from_torchfx(mask_graph))
        lower_output.is_mask_mod_code = "True"
    
    # TODO: infer mask logic
    if infer_mask:
        block_M = int(tune_output.block_M)
        block_N = int(tune_output.block_N)
        import torch
        if block_mask is not None:
            block_mask = create_block_mask(block_mask, Batch, head, seqlen, seqlen, "cuda" if torch.cuda.is_available() else "cpu", block_M, block_N)
        if block_mask is not None:
            lower_output.is_casual = "True" if is_less_causal_mask(block_mask,block_M, block_N) else "False"
        else:
            lower_output.is_casual = "False"
        if block_mask is not None and not is_causal_mask(block_mask, block_M, block_N):
            tlattn_template = TlBlockAttnTemplate
            output_idx_list = [i+1 for i in output_idx_list]
        else:
            tlattn_template = TlAttnTemplate
        
        return tlattn_template(
            TEMPLATE_PATH,
            custom_fwd_inputs=kernel_code_template.input_args,
            custom_fwd_inputs_init=kernel_code_template.alloc,
            final_rowscales_output=kernel_code_template.output_args,
            final_rowscales_save=kernel_code_template.output_args_copy_epilogue,
            custom_fwd_inputs_load_prolog=kernel_code_template.input_args_copy_prologue,
            **lower_custom_inputs_output.__dict__,
            **lower_online_func_output.__dict__,
            **lower_score_mod_output.__dict__,

            **lower_output.__dict__,
            **tune_output.__dict__,
            **tune_output_bwd.__dict__,

            output_idx_list=str(output_idx_list),
            bwd_output_idx_list=str(bwd_output_idx_list)
        )(), block_mask
        
    else:
        tlattn_template = TlAttnTemplate
        lower_output.is_casual = "True" if block_mask is not None else "False"

        return tlattn_template(
            TEMPLATE_PATH,
            custom_fwd_inputs=kernel_code_template.input_args,
            custom_fwd_inputs_init=kernel_code_template.alloc,
            final_rowscales_output=kernel_code_template.output_args,
            final_rowscales_save=kernel_code_template.output_args_copy_epilogue,
            custom_fwd_inputs_load_prolog=kernel_code_template.input_args_copy_prologue,
            **lower_custom_inputs_output.__dict__,
            **lower_online_func_output.__dict__,
            **lower_score_mod_output.__dict__,

            **lower_output.__dict__,
            **tune_output.__dict__,
            **tune_output_bwd.__dict__,

            output_idx_list=str(output_idx_list),
            bwd_output_idx_list=str(bwd_output_idx_list)
        )(), None

