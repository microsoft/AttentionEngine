from ..transform.core import SymbolScalar, SymbolicArray, CustomIO
from ..transform.graph import Var, Const
from ..utils import IndentedCode
from ..codegen.tl_gen import generate_tl_from_dag
from ..template.cute_template import CuteAttnTemplate
from dataclasses import dataclass


@dataclass
class LowerCuteOutput:
    dimqk: str = ""
    dimv: str = ""
    cutlass_dtype: str = ""

    online_rowscales_init: str = ""
    online_rowscales_vardefine: str = ""
    online_fwd_body_vardefine: str = ""
    online_fwd_body: str = ""
    o_scale_var: str = ""
    copy_o_scale_var: str = ""
    copy_online_rowscales: str = ""
    finalize_epilogue_body: str = ""
    finalize_epilogue_body_vardefine: str = ""
    copy_final_rowscales: str = ""
    online_rowscales_0: str = ""
    online_rowscales_0_size: str = "0"
    FrgTensorLSE_type: str = ""

    final_rowscales_params_call: str = ""
    final_rowscales_struct: str = ""
    final_rowscales_store_params_def: str = ""
    final_rowscales_store_code_define: str = ""
    final_rowscales_store_code_assert: str = ""
    final_rowscales_store_code_write: str = ""
    global_ptr_args: str = ""
    global_ptr_params_init: str = ""
    online_rowscale_tensor_def: str = ""
    global_ptr_args_init: str = ""
    global_ptr_args_d: str = ""
    final_rowscale_tensor_fill: str = ""
    final_rowscale_return: str = ""
    final_rowscales_return_struct: str = ""
    final_rowscales_store_code_write_zero: str = ""
    global_ptr_params_def: str = ""
    final_rowscales_params: str = ""

    mainloop_arguments_define: str = ""
    mainloop_params_arg: str = ""
    mainloop_params_arg_input: str = ""
    score_mod_code: str = ""
    global_tensor_args: str = ""
    custom_tensors: str = ""
    
    # TODO: tmp solution
    # void *__restrict__ softmax_lse_ptr; 
    global_ptr_params_def_bwd: str = "void *__restrict__ softmax_lse_ptr;"


def lower_online_func(online_func, lower_cute_output: LowerCuteOutput):
    online_fwd = online_func.online_fwd
    scores = SymbolicArray(
        "scores", Var("scores"), shape_idx=[
            "block_M", "block_N"])
    # online_rowscale_init
    online_rowscales = online_func.online_rowscales
    final_rowscales = online_func.final_rowscales
    b = SymbolScalar("b", Var("b"))
    h = SymbolScalar("h", Var("h"))
    q_idx = SymbolScalar("q_idx", Var("q_idx"))

    # hack: online_rowscale_usecount+1 for codegen
    # for k,v in online_rowscales.items():
    #     online_rowscales[k].count += 1

    # online_fwd
    scores_new, new_online_rowscales, o_scalevar = online_fwd(
        scores, online_rowscales, b, h, q_idx)
    for k, v in new_online_rowscales.items():
        new_online_rowscales[k].count += 1
    # o_scalevar usecount+1
    o_scalevar.count += 1
    # TODO: assert: scores_new==score
    tl_code_online, input_vars_online = generate_tl_from_dag(list(
        new_online_rowscales.values()) + [scores_new, o_scalevar], to_tl=False, to_cute=True)

    # online_fwd_epilogue
    acco = SymbolicArray(
        "acc_o_rowcol",
        Var("acc_o_rowcol"),
        shape_idx=[
            "block_M",
            "dimv"])
    for k, v in online_rowscales.items():
        online_rowscales[k].clear_codegen()
    acco_new, new_final_rowscales\
        = online_func.online_fwd_epilogue(acco, online_rowscales, b, h, q_idx)
    tl_code_online_epilogue, input_vars_online_epilogue = generate_tl_from_dag(
        list(new_final_rowscales.values()) + [acco_new], to_tl=False, to_cute=True)

    # to backend template specific
    online_rowscales_initvalue = ""
    for k, v in online_rowscales.items():
        if v.code.name == "-inf":
            cute_init_value = "-INFINITY"
        else:
            cute_init_value = v.code.name
        online_rowscales_initvalue += f"cute::fill({k}, {cute_init_value});\n"
    lower_cute_output.online_rowscales_init = online_rowscales_initvalue
    lower_cute_output.online_rowscales_vardefine = "\n".join(
        [f"TensorT {var};" for var in online_rowscales.keys()])
    lower_cute_output.online_fwd_body += str(tl_code_online)
    lower_cute_output.online_fwd_body_vardefine += "\n".join(
        [
            f"Tensor {var} = make_fragment_like({list(online_rowscales.keys())[0]});" for var in input_vars_online if (
                var not in online_rowscales.keys() and var != "scores")])  # dict iter on keys
    lower_cute_output.o_scale_var = o_scalevar.varname
    lower_cute_output.copy_o_scale_var = f"cute::copy({o_scalevar.varname}, scores_scale);"
    lower_cute_output.FrgTensorLSE_type = f"typename FrgTensorLSE, "
    for k, v in new_online_rowscales.items():
        if k != v.varname:
            lower_cute_output.copy_online_rowscales += f"cute::copy({v.varname}, {k});\n"

    lower_cute_output.online_rowscales_vardefine += "\n".join(
        [f"TensorT {var};" for var in new_final_rowscales.keys() if var not in online_rowscales.keys()])
    # maube  buggy because of acc_o_rowcol cannot inplace
    lower_cute_output.finalize_epilogue_body_vardefine = "\n".join(
        [
            f"Tensor {var} = make_fragment_like({list(new_final_rowscales.keys())[0]});" for var in input_vars_online_epilogue if (
                var not in new_final_rowscales.keys() and var != "acc_o_rowcol") and var not in online_rowscales.keys()])
    lower_cute_output.finalize_epilogue_body = str(tl_code_online_epilogue)
    for k, v in new_final_rowscales.items():
        if k != v.varname:
            lower_cute_output.copy_final_rowscales += f"cute::copy({v.varname}, {k});\n"

    lower_cute_output.online_rowscales_0 = list(online_rowscales.keys())[
        0] if online_rowscales else ""
    lower_cute_output.online_rowscales_0_size = f"size({lower_cute_output.online_rowscales_0})"

    lower_cute_output.final_rowscales_params_call = ", ".join(
        [f"softmax.{varname}" for varname in new_final_rowscales.keys()]) + ("," if new_final_rowscales else "")
    for k, v in new_final_rowscales.items():
        lower_cute_output.final_rowscales_struct += f"float* ptr_{k};\n" + \
            f"typename Seqlen_traits::LayoutLseT layout_{k};\n"
        lower_cute_output.final_rowscales_store_code_define += f"Tensor m{k} = make_tensor(make_gmem_ptr(epilogue_params.ptr_{k}), epilogue_params.layout_{k});\n" + \
            f"Tensor g{k} = seqlen_traits_q.get_lse_local_tile_tensor( m{k}, Shape<Int<kBlockM>>{{}}, bidh, bidb)(_, m_block);"
        lower_cute_output.final_rowscales_store_code_assert += f"CUTE_STATIC_ASSERT_V(size({k}) == size(taccOcO_row));"
        lower_cute_output.final_rowscales_store_code_write += \
            f"""
        if (get<1>(taccOcO_row(_0{{}})) == 0) {{
            #pragma unroll
            for (int mi = 0; mi < size({k}); ++mi) {{
                const int row = get<0>(taccOcO_row(mi));
                if (row < seqlen_traits_q.actual_seq_len - m_block * kBlockM) {{ g{k}(row) = {k}(mi); }}
            }}
        }}"""
    lower_cute_output.final_rowscales_store_params_def = ", ".join(
        [f"FrgTensorLSE const& {k}" for k in new_final_rowscales.keys()]) + ("," if new_final_rowscales else "")
    lower_cute_output.global_ptr_args = ", ".join(
        [f"void* softmax_{k}_d" for k in new_final_rowscales.keys()]) + ("," if new_final_rowscales else "")
    # params.softmax_lse_ptr = softmax_lse_d;
    lower_cute_output.global_ptr_params_init = "\n".join(
        [f"params.softmax_{k}_ptr = softmax_{k}_d;" for k in new_final_rowscales.keys()])
    # auto softmax_lse = torch::empty({batch_size, num_heads, seqlen_q},
    # opts.dtype(at::kFloat));
    lower_cute_output.online_rowscale_tensor_def = "\n".join(
        [f"auto softmax_{k} = torch::empty({{batch_size, num_heads, seqlen_q}}, opts.dtype(at::kFloat));" for k in new_final_rowscales.keys()])
    # softmax_lse.data_ptr(),
    lower_cute_output.global_ptr_args_init = ", ".join(
        [f"softmax_{k}.data_ptr()" for k in new_final_rowscales.keys()]) + ("," if new_final_rowscales else "")
    lower_cute_output.global_ptr_args_d = ", ".join(
        [f"softmax_{k}_d" for k in new_final_rowscales.keys()]) + ("," if new_final_rowscales else "")
    # softmax_lse.fill_(std::numeric_limits<float>::infinity());
    # TODO: fill num
    lower_cute_output.final_rowscale_tensor_fill = "\n".join(
        [f"softmax_{k}.fill_(std::numeric_limits<float>::infinity());" for k in new_final_rowscales.keys()])
    # softmax_lse,
    lower_cute_output.final_rowscale_return = ", ".join(
        [f"softmax_{k}" for k in new_final_rowscales.keys()]) + ("," if new_final_rowscales else "")
    # args.ptr_LSE, args.layout_LSE
    lower_cute_output.final_rowscales_return_struct = ", ".join(
        [f"args.ptr_{k}, args.layout_{k}" for k in new_final_rowscales.keys()]) + ("," if new_final_rowscales else "")
    # if (thread_idx < seqlen_traits_q.actual_seq_len - m_block * kBlockM) {
    # gLSE(thread_idx) = -INFINITY; }
    lower_cute_output.final_rowscales_store_code_write_zero = "\n".join(
        [f"if (thread_idx < seqlen_traits_q.actual_seq_len - m_block * kBlockM) {{ g{k}(thread_idx) = -INFINITY; }}" for k in new_final_rowscales.keys()])
    # void * __restrict__ softmax_lse_ptr;
    lower_cute_output.global_ptr_params_def = "\n".join(
        [f"void * __restrict__ softmax_{k}_ptr;" for k in new_final_rowscales.keys()])
    # static_cast<float*>(params.softmax_lse_ptr),
    #     // seqlen_traits_q.get_lse_gmem_layout(
    #     //     params.seqlen_q, params.h, params.b
    #     // )  // layout_LSE
    lower_cute_output.final_rowscales_params = ", ".join(
        [f"static_cast<float*>(params.softmax_{k}_ptr), seqlen_traits_q.get_lse_gmem_layout( params.seqlen_q, params.h, params.b)" for k in new_final_rowscales.keys()])  # + ("," if new_final_rowscales else "")

    # print(lower_cute_output.online_rowscales_vardefine)
    # print(lower_cute_output.online_fwd_body)
    # print(lower_cute_output.online_fwd_body_vardefine)
    # print(lower_cute_output.o_scale_var)
    # print(lower_cute_output.copy_online_rowscales)
    # print(lower_cute_output.finalize_epilogue_body)
    # print(lower_cute_output.finalize_epilogue_body_vardefine)
    # print(lower_cute_output.copy_final_rowscales)


def lower_score_mod(score_mod, custom_fwd_inputs,
                    lower_cute_output: LowerCuteOutput):

    b = SymbolScalar("b", Var("b"))
    h = SymbolScalar("h", Var("h"))
    q_idx = SymbolScalar("q_idx", Var("q_idx"))
    kv_idx = SymbolScalar("kv_idx", Var("kv_idx"))
    scores = SymbolicArray(
        "scores", Var("scores"), shape_idx=[
            "block_M", "block_N"])

    scores_new = score_mod(scores, custom_fwd_inputs, b, h, q_idx, kv_idx)
    # TODO: scores_new == scores
    tl_code_score_mod, input_vars_score_mod = generate_tl_from_dag(
        [scores_new], to_tl=False, to_cute=True)

    # lower_cute_output.online_fwd_body += str(tl_code_score_mod)
    # lower_cute_output.online_fwd_body_vardefine += "\n".join([f"Tensor {var}
    # =
    # make_fragment_like({list(custom_fwd_inputs.input_tensors.keys())[0]});"
    # for var in input_vars_score_mod if var not in
    # custom_fwd_inputs.input_tensors.keys()]) # dict iter on keys
    for k, v in custom_fwd_inputs.input_tensors.items():
        if len(v.shape_idx) == 1 and v.shape_idx[0] == "1":
            lower_cute_output.mainloop_arguments_define += f"float const {k};"
            lower_cute_output.global_ptr_params_def += f"float {k};"
            lower_cute_output.score_mod_code += f"const float {k} = mainloop_params.{k};\n"
            lower_cute_output.global_ptr_args += f"float {k}, "
            lower_cute_output.global_ptr_args_d += f"{k}, "
            lower_cute_output.global_tensor_args += f"const float {k}, "
            lower_cute_output.global_ptr_args_init += f"{k}, "
        lower_cute_output.mainloop_params_arg += f"args.{k}, "
        lower_cute_output.mainloop_params_arg_input += f"params.{k}, "
        lower_cute_output.global_ptr_params_init += f"params.{k} = {k};"
        lower_cute_output.custom_tensors += f"{k}, "
    lower_cute_output.score_mod_code += str(tl_code_score_mod)


def lower_cute(score_mod, block_mask, online_func,
               custom_fwd_inputs,
               dimqk, dimv, cutlass_dtype, template_dir=None, output_dir=None):

    lower_cute_output = LowerCuteOutput()
    lower_cute_output.dimqk = str(dimqk)
    lower_cute_output.dimv = str(dimv)
    lower_cute_output.cutlass_dtype = cutlass_dtype

    if score_mod:  # score_mod first
        lower_score_mod(score_mod, custom_fwd_inputs, lower_cute_output)
    if online_func:
        lower_online_func(online_func, lower_cute_output)
        
    # TODO: tmp solution
    if "softmax_lse_ptr" in lower_cute_output.global_ptr_params_def:
        lower_cute_output.global_ptr_params_def_bwd = ""

    if template_dir is not None:
        lower_cute_output.template_dir = template_dir
    if output_dir is not None:
        lower_cute_output.output_dir = output_dir
    return CuteAttnTemplate(
        **lower_cute_output.__dict__,
    )()
