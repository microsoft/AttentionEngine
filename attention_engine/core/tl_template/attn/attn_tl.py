# TL_IMPORT = """
import torch
from tvm import tl
import tvm.tl.language as T

# TL_GLOBAL_FUNC = """
def fast_tanh(A, B):
    return T.call_extern("handle", "fasttanh", T.address_of(A), T.address_of(B))

def make_dq_layout(dQ):
    # atomicAdd can not be vectorized, so we need to reorder dq to match the 8x8 gemm fragment
    return T.Layout(
        dQ.shape, lambda b, l, h, d: [b, l // 8, h, d // 8, (d % 2), 4 * (l % 8) + (d % 8) // 2]
    )

# TL_KERNEL = """
def kernel(batch, heads, seq_len, dim, dimv, 
        block_M = None, block_N = None, num_stages = None, thread_num = None,
        shared_fuse = None):
    # scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e) # 0.69314718  loge(2)
    shape = [batch, seq_len, heads, dim]
    shape_v = [batch, seq_len, heads, dimv]
    # TODO: seqlenkv
    seq_len_kv = seq_len
    dtype = "{{tl_dtype}}" # "float16"
    accum_dtype = "float"
    
    # TODO: mask
    is_casual = {{is_inf_mask}} # True
    # shared_fuse = True

# TL_MAIN = """
    @T.macro
    def score_mod(
        # scores: T.Buffer([block_M, block_N], accum_dtype),
        {{score_mod_inputs | indent(8)}}
        ):
        {{score_mod_body | indent(8)}}
        pass
    
    @T.macro
    def online_func(
        # scores: T.Buffer([block_M, block_N], accum_dtype),
        {{online_func_inputs | indent(8)}}
    ):
        {{online_func_body | indent(8)}}
        pass

        
    @T.prim_func
    def main(
        Q: T.Buffer(shape, dtype), # type: ignore
        K: T.Buffer(shape, dtype), # type: ignore
        V: T.Buffer(shape_v, dtype), # type: ignore
        {{custom_fwd_inputs | indent(8)}}

        Output: T.Buffer(shape_v, dtype), # type: ignore
        {{final_rowscales_output | indent(8)}}
    ):
        with T.Kernel(T.ceildiv(seq_len, block_M), heads, batch, threads=thread_num) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_M, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_N, dimv], dtype)
            # V_local = T.alloc_fragment([block_N, dimv], dtype)
            scores = T.alloc_fragment([block_M, block_N], accum_dtype)
            scores_shared = T.alloc_shared([block_M, block_N], accum_dtype)
            scores_1 = T.alloc_fragment([block_M, block_N], accum_dtype)
            acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
            acc_s_cast_1 = T.alloc_fragment([block_M, block_N], dtype)
            # acc_o = T.alloc_fragment([block_M, dimv], accum_dtype)

            {{custom_fwd_inputs_init | indent(12)}}
            {{online_func_init | indent(12)}}
            {{final_rowscales_init | indent(12)}}

            T.annotate_layout({
                Q_shared: tl.layout.make_swizzled_layout(Q_shared),
                scores_shared: tl.layout.make_swizzled_layout(scores_shared),
                {{swizzle_shared | indent(16)}}
            })
            T.copy(Q[bz, bx * block_M : (bx + 1) * block_M, by, :], Q_shared)
            {{custom_fwd_inputs_load_prolog | indent(12)}}
            T.fill(acc_o, 0)
            T.fill({{o_scale_varname}}, 1.0)

            {{online_rowscales_initvalue | indent(12)}}

            # TODO: mask
            loop_range = (
                T.ceildiv((bx + 1) * block_M, block_N) if is_casual else T.ceildiv(seq_len, block_N)
            )

            for k in T.Pipelined(loop_range, num_stages=num_stages):
                T.copy(K[bz, k * block_N : (k + 1) * block_N, by, :], K_shared)

                # TODO: copy custom_fwd_input_tensor in score_mod&online_func
                {{custom_fwd_inputs_load_shared | indent(16)}}

                # TODO: naive solution: if reduce_max, -T.inf; if reduce_sum, 0
                if is_casual and {{is_inf_mask}}:
                    for i, j in T.Parallel(block_M, block_N):
                        scores[i, j] = T.if_then_else(
                            bx * block_M + i >= k * block_N + j, 0, -T.infinity(scores.dtype)
                        )
                else:
                    T.clear(scores)
                
                T.gemm(Q_shared, K_shared, scores, transpose_B=True, policy= (T.GemmWarpPolicy.FullRow if (not shared_fuse) else T.GemmWarpPolicy.FullCol))
                T.copy(V[bz, k * block_N : (k + 1) * block_N, by, :], V_shared)
                    
                {{custom_fwd_inputs_load_s2r | indent(16)}}
                # call score_mod
                score_mod({{score_mod_inputs_list}}) # scores
                    
                # call online_func
                if shared_fuse:
                    T.copy(scores, scores_shared)
                    T.copy(scores_shared, scores_1)
                    online_func({{online_func_inputs_list}})
                    T.copy(scores_1, acc_s_cast_1)

                else:
                    online_func({{online_func_inputs_list}}) # scores
                    T.copy(scores, acc_s_cast)

                # for i, j in T.Parallel(block_M, dimv):
                #     acc_o[i, j] *= o_scale[i]
                {{o_scale | indent(16)}}
                
                # update online_rowscales
                {{online_rowscales_update | indent(16)}}

                if shared_fuse:
                    T.gemm(acc_s_cast_1, V_shared, acc_o, policy=(T.GemmWarpPolicy.FullCol))
                else:
                    T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
            
            # online_fwd_epilogue
            {{online_func_epilogue | indent(12)}}

            T.copy(acc_o, Output[bz, bx * block_M : (bx + 1) * block_M, by, :])

            # save final_rowscale
            {{final_rowscales_save | indent(12)}}
        
    return main

# TL_KERNEL_BWD_DOO = """
def flashattn_bwd_preprocess(batch, heads, seq_len, dim, dimv):
    dtype = "{{tl_dtype}}" # "float16"
    accum_dtype = "float"
    shape = [batch, seq_len, heads, dim]
    shape_v = [batch, seq_len, heads, dimv]
    blk = 32

    @T.prim_func
    def flash_bwd_prep(
        O: T.Buffer(shape_v, dtype), # type: ignore
        dO: T.Buffer(shape_v, dtype), # type: ignore
        Delta: T.Buffer([batch, heads, seq_len], accum_dtype), # type: ignore
    ):
        with T.Kernel(heads, T.ceildiv(seq_len, blk), batch) as (bx, by, bz):
            o = T.alloc_fragment([blk, blk], dtype)
            do = T.alloc_fragment([blk, blk], dtype)
            acc = T.alloc_fragment([blk, blk], accum_dtype)
            delta = T.alloc_fragment([blk], accum_dtype)
            T.clear(acc)
            for k in range(T.ceildiv(dimv, blk)):
                T.copy(O[bz, by * blk : (by + 1) * blk, bx, k * blk : (k + 1) * blk], o)
                T.copy(dO[bz, by * blk : (by + 1) * blk, bx, k * blk : (k + 1) * blk], do)
                for i, j in T.Parallel(blk, blk):
                    acc[i, j] += o[i, j] * do[i, j]
            T.reduce_sum(acc, delta, 1)
            T.copy(delta, Delta[bz, bx, by * blk : (by + 1) * blk])

    return flash_bwd_prep

# TL_KERNEL_BWD = """
def flashattn_bwd(batch, heads, seq_len, dim, dimv, is_casual, 
                block_M, block_N, thread_num = 128*2):
    sm_scale = (1.0 / dim) ** 0.5
    scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
    shape = [batch, seq_len, heads, dim]
    shape_v = [batch, seq_len, heads, dimv]
    # TODO: seqlenkv
    seq_len_kv = seq_len
    dtype = "{{tl_dtype}}" # "float16"
    accum_dtype = "float"

# TL_MAIN_BWD = """
    @T.macro
    def score_mod(
        # scores: T.Buffer([block_M, block_N], accum_dtype),
        {{score_mod_fwd_inputs | indent(8)}}
        ):
        {{score_mod_fwd_body | indent(8)}}
        pass
    
    @T.macro
    def score_mod_backward(
        # scores: T.Buffer([block_M, block_N], accum_dtype),
        {{score_mod_bwd_inputs | indent(8)}}
    ):
        {{score_mod_backward | indent(8)}}
        pass

    @T.prim_func
    def flash_bwd(
        Q: T.Buffer(shape, dtype), # type: ignore
        K: T.Buffer(shape, dtype), # type: ignore
        V: T.Buffer(shape_v, dtype), # type: ignore
        dO: T.Buffer(shape_v, dtype), # type: ignore

        # custom_fwd_inputs score_mod
        {{custom_fwd_inputs | indent(8)}}

        # final_rowscales
        {{final_rowscales_output | indent(8)}}

        # custom_bwd_inputs
        {{custom_bwd_inputs | indent(8)}}

        dQ: T.Buffer(shape, accum_dtype), # type: ignore
        dK: T.Buffer(shape, dtype), # type: ignore
        dV: T.Buffer(shape_v, dtype), # type: ignore
    ):
        with T.Kernel(heads, T.ceildiv(seq_len, block_M), batch, threads=thread_num) as (bx, by, bz):
            K_shared = T.alloc_shared([block_M, dim], dtype)
            dsT_shared = T.alloc_shared([block_M, block_N], dtype)
            # should not store K to local if dim is large
            # K_local = T.alloc_fragment([block_M, dim], dtype)
            # H100 wgmma
            # K_local_T = T.alloc_fragment([block_M, dim], dtype)
            # V_local = T.alloc_fragment([block_M, dimv], dtype)
            q = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_M, dimv], dtype)
            # qkT = T.alloc_fragment([block_M, block_N], accum_dtype)
            # dsT = T.alloc_fragment([block_M, block_N], accum_dtype)
            qkT_cast = T.alloc_fragment([block_M, block_N], dtype)
            dsT_cast = T.alloc_fragment([block_M, block_N], dtype)

            # final_rowscales_declare
            {{final_rowscales_shared_init | indent(12)}}

            # custom_bwd_declare
            {{custom_bwd_inputs_init | indent(12)}}

            # score_mod_declare
            {{score_mod_bwd_inputs_declare | indent(12)}}
            # score_mod_declare_shared
            {{score_mod_bwd_inputs_declare_shared | indent(12)}}
            
            do = T.alloc_shared([block_N, dimv], dtype)
            dv = T.alloc_fragment([block_M, dimv], accum_dtype)
            dk = T.alloc_fragment([block_M, dim], accum_dtype)
            dq = T.alloc_fragment([block_N, dim], accum_dtype)
            dv_shared = T.alloc_shared([block_N, dimv], dtype)
            dk_shared = T.alloc_shared([block_N, dim], dtype)
            T.annotate_layout(
                {
                    dQ: make_dq_layout(dQ),
                    K_shared: tl.layout.make_swizzled_layout(K_shared),
                    dv_shared: tl.layout.make_swizzled_layout(dv_shared),
                    dk_shared: tl.layout.make_swizzled_layout(dk_shared),
                }
            )
            T.copy(K[bz, by * block_M : (by + 1) * block_M, bx, :], K_shared)
            T.copy(V[bz, by * block_M : (by + 1) * block_M, bx, :], V_shared)
            # T.copy(K_shared, K_local)
            # T.copy(K_shared, K_local_T)
            # custom_fwd_inputs_load_prolog
            {{custom_fwd_inputs_load_prolog | indent(12)}}
            T.clear(dv)
            T.clear(dk)

            # TODO: is causal
            loop_st = T.floordiv(by * block_M, block_N) if is_casual else 0
            loop_ed = T.ceildiv(seq_len, block_N)

            for k in T.Pipelined(loop_st, loop_ed, num_stages=2):
                T.copy(Q[bz, k * block_N : (k + 1) * block_N, bx, :], q)
                {{custom_fwd_inputs_load_shared_bwd | indent(16)}}
                T.clear(qkT)
                T.gemm(K_shared, q, qkT, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                
                # score_mod
                score_mod({{score_mod_inputs_bwd_list}}) # qkT,

                # final_rowscales_load
                {{final_rowscales_load | indent(16)}}

                # online_func_fwd
                {{ online_func_fwd | indent(16) }}
                
                # TODO: is causal
                if is_casual:
                    for i, j in T.Parallel(block_M, block_N):
                        {{score_mod_output_var}}[i, j] = T.if_then_else(
                            by * block_M + i <= k * block_N + j, {{score_mod_output_var}}[i, j], 0
                        )
                
                T.copy(dO[bz, k * block_N : (k + 1) * block_N, bx, :], do)
                T.clear(dsT)
                T.gemm(V_shared, do, dsT, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                
                T.copy({{score_mod_output_var}}, qkT_cast)
                T.gemm(qkT_cast, do, dv, policy=T.GemmWarpPolicy.FullRow)

                # custom_bwd_inputs_load
                {{custom_bwd_inputs_load | indent(16)}}

                # T.clear(dsT)
                # T.gemm(V_local, do, dsT, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                if is_casual:
                    for i, j in T.Parallel(block_M, block_N):
                        dsT[i, j] = T.if_then_else(
                            by * block_M + i <= k * block_N + j, dsT[i, j], 0
                        )

                # custom_bwd
                {{custom_bwd_body | indent(16)}}
                
                # score_mod_backward
                score_mod_backward({{score_mod_bwd_inputs_list}}) #  qkT, 
                  
                                
                T.copy(dsT, dsT_cast)
                T.gemm(dsT_cast, q, dk, policy=T.GemmWarpPolicy.FullRow)

                T.copy(dsT_cast, dsT_shared)
                T.clear(dq)
                # T.gemm(dsT_shared, K_local_T, dq, transpose_A=True)
                T.gemm(dsT_shared, K_shared, dq, transpose_A=True, policy=T.GemmWarpPolicy.FullCol)
                for i, j in T.Parallel(block_N, dim):
                    if k * block_N + i < seq_len:
                        T.atomic_add(dQ[bz, k * block_N + i, bx, j], dq[i, j])
            T.copy(dv, dV[bz, by * block_M : (by + 1) * block_M, bx, :])
            T.copy(dk, dK[bz, by * block_M : (by + 1) * block_M, bx, :])

    return flash_bwd              

# TL_KERNEL_BWD_POSTPROCESS = """
def flashattn_bwd_postprocess(batch, heads, seq_len, dim, dimv):
    dtype = "{{tl_dtype}}" # "float16"
    accum_dtype = "float"
    shape = [batch, seq_len, heads, dim]
    shape_v = [batch, seq_len, heads, dimv]
    blk = 64

    @T.prim_func
    def flash_bwd_post(
        dQ: T.Buffer(shape, accum_dtype), # type: ignore
        dQ_out: T.Buffer(shape, dtype), # type: ignore
    ):
        with T.Kernel(T.ceildiv(seq_len, blk), heads, batch, threads=128) as (bx, by, bz):
            T.annotate_layout({dQ: make_dq_layout(dQ)})
            T.copy(
                dQ[bz, bx * blk : (bx + 1) * blk, by, :],
                dQ_out[bz, bx * blk : (bx + 1) * blk, by, :],
            )

    return flash_bwd_post

# TL_INFERFACE = """
class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, *custom_fwd_inputs):
        BATCH, N_CTX, H, D_HEAD = q.shape
        D_HEADV = v.shape[-1]
        block_M = {{block_M}} # 128
        block_N = {{block_N}} # 128 if D_HEAD <= 128 else 64
        stages = {{stages}} # 2
        thread_num = {{thread_num}} # 256
        shared_fuse = {{shared_fuse}} # False
        output_idx_list = {{output_idx_list}}
        mod = tl.cached(kernel, output_idx_list, BATCH, H, N_CTX, D_HEAD, D_HEADV, block_M, block_N, stages, thread_num, shared_fuse)
        if len(output_idx_list) == 1:
            o = mod(q, k, v, *custom_fwd_inputs)
            final_scale = []
        else:
            o, *final_scale = mod(q, k, v, *custom_fwd_inputs)
        ctx.save_for_backward(q, k, v, o, *custom_fwd_inputs, *final_scale)
        return o
    
    @staticmethod
    def backward(ctx, do):
        q, k, v, o, *tmp = ctx.saved_tensors
        BATCH, N_CTX, H, D_HEAD = q.shape
        D_HEAD_V = v.shape[-1]
        # custom_fwd_inputs = tmp[:-{{final_rowscales_length}}]
        # final_rowscales = tmp[-{{final_rowscales_length}}:]
        maybe_contiguous = lambda x: x.contiguous() if x.stride(-1) != 1 else x
        do, q, k, v, o = [maybe_contiguous(x) for x in (do, q, k, v, o)]
        block_M = {{block_M_bwd}} # 128
        block_N = {{block_N_bwd}} # 64 
        thread_num = {{thread_num_bwd}} # 256
        mod_prep = tl.cached(flashattn_bwd_preprocess, [2], BATCH, H, N_CTX, D_HEAD, D_HEAD_V)
        mod_post = tl.cached(flashattn_bwd_postprocess, [1], BATCH, H, N_CTX, D_HEAD, D_HEAD_V)
        if {{isused_doosum}}:
            delta = mod_prep(o, do)
        # TODO: causal
        is_casual = {{is_inf_mask}}
        output_idx_list = {{bwd_output_idx_list}}
        mod = tl.cached(
            flashattn_bwd, output_idx_list, BATCH, H, N_CTX, D_HEAD, D_HEAD_V, is_casual, block_M, block_N, thread_num
        )
        if {{isused_doosum}}:
            dq, dk, dv = mod(q, k, v, do, *tmp, delta)
        else:
            dq, dk, dv = mod(q, k, v, do, *tmp)
        dq = mod_post(dq)
        none_list = [None] * len(tmp)
        return dq, dk, dv, *none_list

attention = _attention.apply


