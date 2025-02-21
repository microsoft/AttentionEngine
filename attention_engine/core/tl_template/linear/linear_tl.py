import torch
import tilelang as tl
import tilelang.language as T

import triton.language as triton_lang
import triton

import itertools

import einops

# ---------------------TRITON_KERNEL_CUMSUM
@triton.jit
def chunk_local_cumsum_scalar_kernel(
    s,
    o,
    T: triton_lang.constexpr,
    BT: triton_lang.constexpr,
):
    i_t, i_bh = triton_lang.program_id(0), triton_lang.program_id(1)
    p_s = triton_lang.make_block_ptr(s + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    p_o = triton_lang.make_block_ptr(o + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    # [BT, BS]
    b_s = triton_lang.load(p_s, boundary_check=(0,)).to(triton_lang.float32)
    b_o = triton_lang.cumsum(b_s, axis=0)
    triton_lang.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0,))

def chunk_local_cumsum_scalar(g, BT):
    B, H, T = g.shape
    NT = triton.cdiv(T, BT)
    g_org, g = g, torch.empty_like(g, dtype=torch.float)
    grid = (NT, B * H)
    chunk_local_cumsum_scalar_kernel[grid](
        g_org, g,
        T=T, BT=BT
    )
    return g

# ---------------------TRITON_KERNEL_REV_CUMSUM
@triton.jit
def compute_final_dg(
    dg,
    o,
    T: triton_lang.constexpr,
    BT: triton_lang.constexpr,
):
    i_t, i_bh = triton_lang.program_id(0), triton_lang.program_id(1)

    p_o = triton_lang.make_block_ptr(dg + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    b_o = triton_lang.load(p_o, boundary_check=(0,))
    b_o = b_o - triton_lang.cumsum(b_o, axis=0) + triton_lang.sum(b_o, axis=0)
    p_o = triton_lang.make_block_ptr(o + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    triton_lang.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0,))


# --------------- TL_KERNEL_H
def chunk_fwd_h(
        batch, headq, headk, head, seqlen, dim, dimv,
        BT, BK, BV, num_stages, num_threads
):
    # BT = 64
    # BK = 64
    # BV = 64
    NT = seqlen // BT
    NK = dim // BK
    NV = dimv // BV
    dtype = "bfloat16"
    accum_dtype = "float"
    num_stages = num_stages
    LOG2E = 1.44269504
    
    seq_len = seqlen
    heads = head
    dimqk = dim

    assert(head % headk == 0)
    head_headk_ratio = head // headk

    @T.prim_func
    def main(
        k: T.Buffer((batch, headk, seqlen, dim), dtype), # type: ignore
        v: T.Buffer((batch, head, seqlen, dimv), dtype), # type: ignore
        g: T.Buffer((batch, head, seqlen), accum_dtype), # type: ignore
        {{chunk_h_custom_inputs_list | indent(8)}}
        h: T.Buffer((batch, head, NT*dim, dimv), dtype), # type: ignore
    ):
        with T.Kernel(NK, NV, batch * head, threads=num_threads) as (bx, by, bz):

            b_h = T.alloc_fragment((BK, BV), accum_dtype)
            # b_h_cast = T.alloc_fragment((BK, BV), dtype)
            b_h_shared = T.alloc_shared((BK, BV), dtype)
            b_k = T.alloc_fragment((BT, BK), dtype)
            b_k_shared = T.alloc_shared((BT, BK), dtype)
            b_kt = T.alloc_fragment((BK, BT), dtype)
            b_v_shared = T.alloc_shared((BT, BV), dtype)
            b_v = T.alloc_fragment((BT, BV), dtype)
            b_g = T.alloc_fragment((BT), accum_dtype)
            b_g_shared = T.alloc_shared((BT), accum_dtype, scope="shared")
            b_glast = T.alloc_fragment((1), accum_dtype)
            {{h_alloc_buffer_list | indent(12)}}

            bhead = bz % head
            bb = bz // head

            bheadk = bhead // head_headk_ratio
            
            T.annotate_layout({
                # b_v_shared: tl.layout.make_swizzled_layout(b_v_shared),
                b_k_shared: tl.layout.make_swizzled_layout(b_k_shared),
                b_h_shared: tl.layout.make_swizzled_layout(b_h_shared),
            }
            )
            T.clear(b_h)

            for i_t in T.Pipelined(NT, num_stages=num_stages):
                # T.copy(b_h, b_h_shared)
                # T.copy(h[bb,bhead,(i_t*dim+bx*BK):(i_t*dim+(bx+1)*BK), by*BV:(by+1)*BV], b_h)
                # T.copy(k[bb,bhead,i_t*BT:(i_t+1)*BT,bx*BK:(bx+1)*BK], b_k)
                T.copy(k[bb,bheadk,i_t*BT:(i_t+1)*BT,bx*BK:(bx+1)*BK], b_k_shared)
                T.copy(v[bb,bhead,i_t*BT:(i_t+1)*BT,by*BV:(by+1)*BV], b_v_shared)
                T.copy(g[bb,bhead,i_t*BT:(i_t+1)*BT], b_g_shared)
                # T.copy(v[bb,bhead,i_t*BT:(i_t+1)*BT,by*BV:(by+1)*BV], b_v)
                # T.copy(b_v, b_v_shared)
                # T.copy(g[bb,bhead,i_t*BT:(i_t+1)*BT], b_g)
                # T.copy(g[bb,bhead,(i_t+1)*BT-1:(i_t+1)*BT], b_glast)
                b_glast[0] = g[bb,bhead,(i_t+1)*BT-1]

                T.copy(b_h, b_h_shared) # implicit cast
                T.copy(b_h_shared, h[bb,bhead,i_t*dim+bx*BK:(i_t)*dim+(bx+1)*BK,by*BV:(by+1)*BV])
                # T.copy(b_h, h[bb,bhead,i_t*dim+bx*BK:(i_t)*dim+(bx+1)*BK,by*BV:(by+1)*BV])

                # scalar_decay
                # for i0 in T.Parallel(BT):
                #     b_g[i0] = T.exp(b_g[i0])

                for i0, i1 in T.Parallel(BK, BV):
                    b_h[i0, i1] = b_h[i0, i1] * T.exp2(b_glast[0] * LOG2E)
                
                # for i0,i1 in T.Parallel(BT, BV):
                #     b_v_shared[i0,i1] *= T.exp2((b_glast[0] - b_g[i0]) * 1.44269504)

                T.copy(b_k_shared, b_k)
                
                {{k_mod_expr_fused_h | indent(16)}}
                
                T.copy(b_g_shared, b_g)
                for i0, i1 in T.Parallel(BK, BT):
                    b_kt[i0, i1] = b_k[i1, i0]*T.exp2((b_glast[0] - b_g[i1]) * LOG2E)

                T.gemm(b_kt, b_v_shared, b_h, transpose_B=False)
                # T.copy(b_h, b_h_shared)
                
                # TODO
                # T.copy(b_h, h[bb,bhead,i_t*dim+bx*BK:(i_t)*dim+(bx+1)*BK,by*BV:(by+1)*BV])
    
    return main


# --------------- TL_KERNEL_O 
def chunk_o(
        batch, headq,headk, head, seqlen, dim, dimv,
        BT, BK, BV, num_stages, num_threads
):
    # BT = 64
    # BK = 64
    # BV = 64
    NT = seqlen // BT
    NK = dim // BK
    NV = dimv // BV
    dtype = "bfloat16"
    accum_dtype = "float"
    num_stages = num_stages
    LOG2E = 1.44269504

    scale = 1.0
    seq_len = seqlen
    heads = head
    dimqk = dim

    assert(head % headk == 0)
    head_headk_ratio = head // headk
    assert(head % headq == 0)
    head_headq_ratio = head // headq

    @T.prim_func
    def main(
        h: T.Buffer((batch,head,NT*dim,dimv), dtype), # type: ignore
        q: T.Buffer((batch,headq,seqlen,dim), dtype), # type: ignore
        k: T.Buffer((batch,headk,seqlen,dim), dtype), # type: ignore
        v: T.Buffer((batch,head,seqlen,dimv), dtype), # type: ignore
        g: T.Buffer((batch,head,seqlen), accum_dtype), # type: ignore
        {{chunk_o_custom_inputs_list | indent(8)}}
        o: T.Buffer((batch,head,seqlen,dimv), dtype), # type: ignore
        # custom fwd inputs
    ):
        with T.Kernel(NV, NT, batch * head, threads=num_threads) as (bx, by, bz):
            bo = T.alloc_fragment((BT, BV), dtype=accum_dtype)
            bo_shared = T.alloc_shared((BT, BV), dtype=dtype)
            bs = T.alloc_fragment((BT, BT), dtype=accum_dtype)
            bs_cast = T.alloc_fragment((BT, BT), dtype=dtype)
            bq = T.alloc_fragment((BT, BK), dtype=dtype)
            bq_shared = T.alloc_shared((BT,BK), dtype=dtype)
            bk_shared = T.alloc_shared((BT,BK), dtype=dtype)
            bv_shared = T.alloc_shared((BT,BV), dtype=dtype)
            b_state_shared = T.alloc_shared((BK, BV), dtype=dtype)
            bg_shared = T.alloc_shared((BT,), dtype=accum_dtype, scope = "shared")
            bg = T.alloc_fragment((BT,), dtype=accum_dtype)
            bg1 = T.alloc_fragment((BT,), dtype=accum_dtype)
            {{o_alloc_buffer_list | indent(12)}}
            # custom fwd inputs init

            bb = bz // head
            bh = bz % head

            bhk = bh // head_headk_ratio
            bhq = bh // head_headq_ratio

            T.annotate_layout({
                bq_shared: tl.layout.make_swizzled_layout(bq_shared),
                bo_shared: tl.layout.make_swizzled_layout(bo_shared),
            })
            T.clear(bo)
            T.clear(bs)
            for ik in T.Pipelined(NK, num_stages=num_stages):
                # pipeline here
                T.copy(q[bb, bhq, by*BT:(by+1)*BT, ik*BK:(ik+1)*BK], bq_shared)
                # T.copy(q[bb, bh, by*BT:(by+1)*BT, ik*BK:(ik+1)*BK], bq)
                T.copy(k[bb, bhk, by*BT:(by+1)*BT, ik*BK:(ik+1)*BK], bk_shared)

                T.copy(h[bb, bh, by*dim+ik*BK:by*dim+(ik+1)*BK, bx*BV:(bx+1)*BV], b_state_shared)
                
                T.copy(bq_shared, bq)
                # q_mod here (fused)
                {{q_mod_expr | indent(16)}}

                T.gemm(bq, bk_shared, bs, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.gemm(bq, b_state_shared, bo, transpose_B=False, policy=T.GemmWarpPolicy.FullRow)
            
            # T.copy(g[bb, bh, by*BT:(by+1)*BT], bg)
            # T.copy(g[bb, bh, by*BT:(by+1)*BT], bg1)
            T.copy(g[bb, bh, by*BT:(by+1)*BT], bg_shared)
            T.copy(bg_shared, bg)
            T.copy(bg_shared, bg1)
            for i0,i1 in T.Parallel(BT,BV):
                bo[i0,i1] *= T.exp2(bg[i0] * LOG2E)
            
            for i0,i1 in T.Parallel(BT,BT):
                bs[i0,i1] *= T.exp2((bg[i0]-bg1[i1]) * LOG2E)
            # fix nan bug
            for i0,i1 in T.Parallel(BT,BT):
                bs[i0,i1] = T.if_then_else(
                    i0 >= i1, bs[i0,i1], 0.0
                )

            # v_mod here (fused)
            {{v_mod_expr_fused_o | indent(12)}}
            
            T.copy(v[bb, bh, by*BT:(by+1)*BT, bx*BV:(bx+1)*BV], bv_shared)
            T.copy(bs, bs_cast)
            T.gemm(bs_cast, bv_shared, bo, policy=T.GemmWarpPolicy.FullRow)
            # T.copy(bo, o[bb, bh, by*BT:(by+1)*BT, bx*BV:(bx+1)*BV]) # slow for stride between thread
            T.copy(bo, bo_shared) # implicit type convert
            T.copy(bo_shared, o[bb, bh, by*BT:(by+1)*BT, bx*BV:(bx+1)*BV])
    
    return main

# --------------- TL_KERNEL_BWD_dh
def chunk_bwd_kernel_dh(
    batch, headq, headk, head, seqlen, dim, dimv,
    BT, BK, BV, num_stages = 1, thread_num = 128
):
    NT = seqlen // BT
    NK = dim // BK
    NV = dimv // BV
    dtype = "bfloat16"
    accum_dtype = "float"
    num_stages = num_stages
    thread_num = thread_num

    scale = 1.0
    assert(head % headk == 0)
    head_headk_ratio = head // headk
    assert(head % headq == 0)
    head_headq_ratio = head // headq

    @T.prim_func
    def main(
        q: T.Buffer((batch, headq, seqlen, dim), dtype), # type: ignore
        k: T.Buffer((batch, headk, seqlen, dim), dtype), # type: ignore
        v: T.Buffer((batch, head, seqlen, dimv), dtype), # type: ignore
        g: T.Buffer((batch, head, seqlen), accum_dtype), # type: ignore
        do: T.Buffer((batch, head, seqlen, dimv), dtype), # type: ignore
        
        dh: T.Buffer((batch, head, NT*dim, dimv), dtype), # type: ignore
    ):
        with T.Kernel(NK, NV, batch*head, threads=thread_num) as (bx, by, bz):
            b_dh = T.alloc_fragment((BK, BV), accum_dtype)
            dh_shared = T.alloc_shared((BK, BV), dtype)
            q_shared = T.alloc_shared((BT, BK), dtype)
            q_local = T.alloc_fragment((BT, BK), dtype)
            q_local_T = T.alloc_fragment((BK, BT), dtype)
            do_shared = T.alloc_shared((BT, BV), dtype)

            bg_local = T.alloc_fragment((BT), accum_dtype)
            bg_shared = T.alloc_shared((BT), accum_dtype, scope="shared")
            bg_last = T.alloc_fragment((1), accum_dtype)

            bhead = bz % head
            bb = bz // head
            bheadk = bhead // head_headk_ratio
            bheadq = bhead // head_headq_ratio

            T.clear(b_dh)
            loop_st = 0
            loop_ed = NT
            for i_t0 in T.Pipelined(NT, num_stages=num_stages):
                i_t = loop_ed - 1 - i_t0

                # T.copy(b_dh, dh[bb, bhead, (i_t*dim + bx*BK):(i_t*dim + (bx+1)*BK), by*BV:(by+1)*BV]) # implicit cast
                T.copy(b_dh, dh_shared)
                T.copy(dh_shared,  dh[bb, bhead, (i_t*dim + bx*BK):(i_t*dim + (bx+1)*BK), by*BV:(by+1)*BV])

                T.copy(q[bb, bheadq, i_t*BT:(i_t+1)*BT, bx*BK:(bx+1)*BK], q_shared)
                T.copy(do[bb, bhead, i_t*BT:(i_t+1)*BT, by*BV:(by+1)*BV], do_shared)
                
                T.copy(q_shared, q_local)

                T.copy(g[bb, bhead, i_t*BT:(i_t+1)*BT], bg_shared)

                # q_mod 
                for i,j in T.Parallel(BT, BK):
                    q_local[i,j] *= scale

                T.copy(bg_shared, bg_local)
                bg_last[0] = bg_shared[BT-1]
                for i,j in T.Parallel(BK, BT):
                    q_local_T[i,j] = q_local[j,i] * T.exp(bg_local[j])

                for i,j in T.Parallel(BK, BV):
                    b_dh[i,j] *= T.exp(bg_last[0])
                
                T.gemm(q_local_T, do_shared, b_dh, transpose_B=False)
        
    return main

# --------------- TL_KERNEL_BWD_dqkg
def chunk_bwd_dqkg(
    batch, headq, headk, head, seqlen, dim, dimv,
    BT, BK, BV, num_stages = 1, thread_num = 128
):
    NT = seqlen // BT
    NK = dim // BK
    NV = dimv // BV
    dtype = "bfloat16"
    accum_dtype = "float"
    num_stages = num_stages
    thread_num = thread_num

    scale = 1.0
    assert(head % headk == 0)
    head_headk_ratio = head // headk
    assert(head % headq == 0)
    head_headq_ratio = head // headq

    @T.prim_func
    def main(
        q: T.Buffer((batch, headq, seqlen, dim), dtype), # type: ignore
        k: T.Buffer((batch, headk, seqlen, dim), dtype), # type: ignore
        v: T.Buffer((batch, head, seqlen, dimv), dtype), # type: ignore
        h: T.Buffer((batch, head, NT*dim, dimv), dtype), # type: ignore
        g: T.Buffer((batch, head, seqlen), accum_dtype), # type: ignore
        do: T.Buffer((batch, head, seqlen, dimv), dtype), # type: ignore
        dh:  T.Buffer((batch, head, NT*dim, dimv), dtype), # type: ignore

        dq: T.Buffer((batch, head, seqlen, dim), dtype), # type: ignore
        dk: T.Buffer((batch, head, seqlen, dim), dtype), # type: ignore
        dg: T.Buffer((NK, batch, head, seqlen), accum_dtype), # type: ignore
    ):
        with T.Kernel(NK, NT, batch*head, threads=thread_num) as (bx, by, bz):
            b_g = T.alloc_fragment((BT), accum_dtype)
            b_g1 = T.alloc_fragment((BT), accum_dtype)
            b_g_last = T.alloc_fragment((1), accum_dtype)
            b_g_T = T.alloc_fragment((BT), accum_dtype)
            b_g_shared = T.alloc_shared((BT), accum_dtype, scope="shared")
            b_v_shared = T.alloc_shared((BT, BV), dtype)
            b_do_shared = T.alloc_shared((BT, BV), dtype)
            b_dh_shared = T.alloc_shared((BK, BV),dtype)
            b_dh_local = T.alloc_fragment((BK, BV),dtype)
            b_h_shared = T.alloc_shared((BK, BV),dtype)
            b_h_local = T.alloc_fragment((BK, BV),dtype)
            b_hdh = T.alloc_fragment((1,BK*BV),accum_dtype)
            k_shared = T.alloc_shared((BT,BK), dtype)
            q_shared = T.alloc_shared((BT,BK),dtype)
            q_local = T.alloc_fragment((BT,BK),dtype)
            k_local = T.alloc_fragment((BT,BK),dtype)
            k_local1 = T.alloc_fragment((BT,BK),dtype)

            b_dq = T.alloc_fragment((BT, BK), accum_dtype)
            b_dq_shared = T.alloc_shared((BT,BK), dtype)
            dq_shared = T.alloc_shared((BT,BK), accum_dtype)
            b_dq1 = T.alloc_fragment((BT, BK), accum_dtype)
            b_dk = T.alloc_fragment((BT, BK), accum_dtype)
            b_dk_shared = T.alloc_shared((BT,BK), dtype)
            dk_shared = T.alloc_shared((BT,BK), accum_dtype)
            b_dk1 = T.alloc_fragment((BT, BK), accum_dtype)
            b_ds = T.alloc_fragment((BT, BT), accum_dtype)
            b_ds_cast = T.alloc_fragment((BT, BT), dtype)
            b_ds_shared = T.alloc_shared((BT, BT), dtype)
            b_dg_last = T.alloc_fragment((1),accum_dtype)
            b_dg_last_tmp = T.alloc_fragment((1),accum_dtype)
            # b_dg_last_local = T.alloc_local((1),accum_dtype)
            # b_dg_last_shared = T.alloc_shared((1),accum_dtype, scope="shared")
            b_dg_qk = T.alloc_fragment((BT,BK),accum_dtype)
            b_dg = T.alloc_fragment((BT),accum_dtype)

            b_dkk = T.alloc_fragment((1,BT*BK), accum_dtype)
            # b_dkksum = T.alloc_fragment((BT), accum_dtype)
            # b_dkksum_shared = T.alloc_shared((BT), accum_dtype, scope="shared")
            # b_dkksum_T = T.alloc_fragment((1,BT), accum_dtype)

            tx = T.thread_binding(0, thread_num, thread="threadIdx.x")

            bhead = bz % head
            bb = bz // head
            bheadk = bhead // head_headk_ratio
            bheadq = bhead // head_headq_ratio

            T.copy(g[bb, bhead, by*BT:(by+1)*BT], b_g_shared)
            b_g_last[0] = b_g_shared[BT-1]

            T.clear(b_dg_last)
            T.clear(b_dg)
            T.clear(b_ds)
            T.clear(b_dq)
            T.clear(b_dk)
            for i_v in T.Pipelined(NV, num_stages=num_stages):
                T.copy(v[bb,bhead, by*BT:(by+1)*BT, i_v*BV:(i_v+1)*BV], b_v_shared)
                T.copy(do[bb,bhead, by*BT:(by+1)*BT, i_v*BV:(i_v+1)*BV], b_do_shared)

                T.copy(dh[bb,bhead,by*dim+bx*BK:by*dim+(bx+1)*BK, i_v*BV:(i_v+1)*BV], b_dh_shared)
                T.copy(h[bb,bhead,by*dim+bx*BK:by*dim+(bx+1)*BK, i_v*BV:(i_v+1)*BV], b_h_shared)

                T.gemm(b_do_shared, b_v_shared, b_ds, transpose_A=False, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                
                T.copy(b_h_shared, b_h_local)
                T.copy(b_dh_shared, b_dh_local)
                for i,j in T.Parallel(BK,BV):
                    b_hdh[0,i*BV+j] = b_h_local[i,j]* b_dh_local[i,j]
                # tl only support clear=True for sharedmemory reduce
                T.reduce_sum(b_hdh, b_dg_last_tmp,dim=1) # , clear=True)
                b_dg_last[0] += b_dg_last_tmp[0]
                T.gemm(b_do_shared, b_h_shared, b_dq, transpose_A=False, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.gemm(b_v_shared, b_dh_shared, b_dk, transpose_A=False, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

            # T.clear(b_dg_last)
            # for ii in T.serial(NV):
            #     T.copy(dh[bb,bhead,by*dim+bx*BK:by*dim+(bx+1)*BK, ii*BV:(ii+1)*BV], b_dh_local)
            #     T.copy(h[bb,bhead,by*dim+bx*BK:by*dim+(bx+1)*BK, ii*BV:(ii+1)*BV], b_h_local)
            #     for i,j in T.Parallel(BK,BV):
            #         b_hdh[0,i*BV+j] = b_h_local[i,j]* b_dh_local[i,j]
            #     T.reduce_sum(b_hdh, b_dg_last,dim=1, clear=False)
                
            T.copy(k[bb, bheadk, by*BT:(by+1)*BT, bx*BK:(bx+1)*BK], k_shared)
            T.copy(q[bb, bheadq, by*BT:(by+1)*BT, bx*BK:(bx+1)*BK], q_shared)
            
            b_dg_last[0] *= T.exp(b_g_last[0])

            T.copy(b_g_shared, b_g)
            for i,j in T.Parallel(BT,BK):
                b_dq[i,j] *= T.exp(b_g[i])
                # qmod_bwd many place
                b_dq[i,j] *= scale

            # reg fuse
            for i,j in T.Parallel(BT,BK):
                b_dk[i,j] *= T.exp(-b_g[i]+b_g_last[0])
            
            
            # possible accuracy loss
            T.copy(b_g_shared, b_g1)
            T.copy(b_g_shared, b_g_T)
            for i,j in T.Parallel(BT,BT):
                b_ds[i,j] = T.if_then_else(
                    i >= j, b_ds[i,j]*scale*T.exp(b_g1[i]-b_g_T[j]), 0
                )
            T.copy(b_ds,b_ds_cast)
            
            for i,j in T.Parallel(BT,BK):
                b_dkk[0,i*BK+j] = b_dk[i,j]
            T.copy(k_shared, k_local1)
            for i,j in T.Parallel(BT,BK):
                b_dkk[0,i*BK+j] *= k_local1[i,j]
            T.reduce_sum(b_dkk, b_dg_last_tmp, dim=1)# , clear=True)
            b_dg_last[0] += b_dg_last_tmp[0]
            
            T.gemm(b_ds_cast,k_shared,b_dq,transpose_A=False, transpose_B=False,policy=T.GemmWarpPolicy.FullRow)
            T.copy(b_ds_cast, b_ds_shared)
            T.gemm(b_ds_shared, q_shared, b_dk, transpose_A=True,transpose_B=False,policy=T.GemmWarpPolicy.FullRow)
            # implicit cast
            # T.copy(b_dq, dq_shared)
            # T.copy(dq_shared, b_dq1)
            for i,j in T.Parallel(BT,BK):
                b_dg_qk[i,j] = b_dq[i,j]
            T.copy(q_shared, q_local)
            T.copy(k_shared, k_local)
            # T.copy(b_dk, dk_shared)
            # T.copy(dk_shared, b_dk1)
            for i,j in T.Parallel(BT,BK):
                b_dg_qk[i,j] = b_dg_qk[i,j] * q_local[i,j] - b_dk[i,j]*k_local[i,j]

            T.reduce_sum(b_dg_qk,b_dg,dim=1)# ,clear=True)
            
            for i in T.Parallel(BT):
                b_dg[i] = T.if_then_else(
                    i < BT-1, b_dg[i], b_dg[i]+b_dg_last[0]
                )
            T.copy(b_dq, b_dq_shared)
            T.copy(b_dq_shared, dq[bb, bhead, by*BT:(by+1)*BT, bx*BK:(bx+1)*BK])
            # T.copy(b_dq, dq[bb, bhead, by*BT:(by+1)*BT, bx*BK:(bx+1)*BK])
            T.copy(b_dk, b_dk_shared)
            T.copy(b_dk_shared, dk[bb, bhead, by*BT:(by+1)*BT, bx*BK:(bx+1)*BK])
            # T.copy(b_dk, dk[bb, bhead, by*BT:(by+1)*BT, bx*BK:(bx+1)*BK])
            T.copy(b_dg, b_g_shared)
            T.copy(b_g_shared, dg[bx, bb, bhead, by*BT:(by+1)*BT])
            # T.copy(b_dg, dg[bx, bb, bhead, by*BT:(by+1)*BT])

    return main

# --------------- TL_KERNEL_BWD_dv
def chunk_bwd_kernel_dv(
    batch, headq, headk, head, seqlen, dim, dimv,
    BT, BK, BV, num_stages = 1, thread_num = 128
):
    NT = seqlen // BT
    NK = dim // BK
    NV = dimv // BV
    dtype = "bfloat16"
    accum_dtype = "float"
    num_stages = num_stages
    thread_num = thread_num

    scale = 1.0
    assert(head % headk == 0)
    head_headk_ratio = head // headk
    assert(head % headq == 0)
    head_headq_ratio = head // headq

    @T.prim_func
    def main(
        q: T.Buffer((batch, headq, seqlen, dim), dtype), # type: ignore
        k: T.Buffer((batch, headk, seqlen, dim), dtype), # type: ignore
        g: T.Buffer((batch, head, seqlen), accum_dtype), # type: ignore
        do: T.Buffer((batch, head, seqlen, dimv), dtype), # type: ignore
        dh: T.Buffer((batch, head, NT*dim, dimv), dtype), # type: ignore
        
        dv: T.Buffer((batch, head, seqlen, dimv), dtype), # type: ignore
    ):
        with T.Kernel(NV, NT, batch*head, threads=thread_num) as (bx, by, bz):
            b_dv = T.alloc_fragment((BT, BV), accum_dtype)
            b_dv_shared = T.alloc_shared((BT, BV), dtype)
            b_g = T.alloc_fragment((BT), accum_dtype)
            b_g1 = T.alloc_fragment((BT), accum_dtype)
            b_g2 = T.alloc_fragment((BT), accum_dtype)
            b_g_shared = T.alloc_shared((BT), accum_dtype, scope="shared")
            b_g_last = T.alloc_fragment((1), accum_dtype)
            b_A = T.alloc_fragment((BT, BT), accum_dtype)
            b_A_cast = T.alloc_fragment((BT, BT), dtype)
            b_q_shared = T.alloc_shared((BT, BK), dtype)
            b_q_local = T.alloc_fragment((BT, BK), dtype)
            b_k_shared = T.alloc_shared((BT, BK), dtype)
            b_k_local = T.alloc_fragment((BT, BK), dtype)
            b_dh_shared = T.alloc_shared((BK, BV), dtype)
            b_dh_local = T.alloc_fragment((BK, BV), dtype)
            b_do_shared = T.alloc_shared((BT, BV), dtype)
            
            bhead = bz % head
            bb = bz // head
            bheadk = bhead // head_headk_ratio
            bheadq = bhead // head_headq_ratio
            
            T.copy(g[bb, bhead, by*BT:(by+1)*BT], b_g_shared)
            T.copy(b_g_shared, b_g)
            b_g_last[0] = b_g_shared[BT-1]
            
            T.clear(b_dv)
            T.clear(b_A)
            for i_k in T.Pipelined(NK, num_stages=num_stages):
                T.copy(k[bb, bheadk, by*BT:(by+1)*BT, i_k*BK:(i_k+1)*BK], b_k_shared)
                T.copy(dh[bb, bhead, by*dim+i_k*BK:by*dim+(i_k+1)*BK, bx*BV:(bx+1)*BV], b_dh_shared)
                T.copy(q[bb, bheadq, by*BT:(by+1)*BT, i_k*BK:(i_k+1)*BK], b_q_shared)
                
                T.gemm(b_k_shared, b_dh_shared, b_dv)
                T.gemm(b_k_shared, b_q_shared, b_A, transpose_B=True)
            for i,j in T.Parallel(BT, BV):
                b_dv[i,j] *= T.exp(-b_g[i] + b_g_last[0])
            T.copy(b_g_shared, b_g1)
            T.copy(b_g_shared, b_g2)
            for i,j in T.Parallel(BT, BT):
                b_A[i,j] *= T.exp(-b_g1[i] + b_g2[j]) * scale
            for i,j in T.Parallel(BT,BT):
                b_A[i,j] = T.if_then_else(
                    i <= j, b_A[i,j], 0
                )
            
            T.copy(do[bb, bhead, by*BT:(by+1)*BT, bx*BV:(bx+1)*BV], b_do_shared)
            T.copy(b_A,b_A_cast)
            T.gemm(b_A_cast, b_do_shared, b_dv)
            T.copy(b_dv, b_dv_shared)
            T.copy(b_dv_shared, dv[bb, bhead, by*BT:(by+1)*BT, bx*BV:(bx+1)*BV])
   
    return main
 

# --------------- TL_INTERFACE
class LinearAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, decay, {{custom_inputs_list}} ):
        BATCH, HQ, N_CTX, D_HEAD = q.shape
        D_HEADV = v.shape[-1]
        HK = k.shape[1]
        H = v.shape[1]

        # autotuner here
        # BT = 64
        # BK_h = 64
        # BV_h = 64
        # num_stages_h = 2
        # num_threads_h = 128
        # BK_o = 64
        # BV_o = 64
        # num_stages_o = 2
        # num_threads_o = 128
        BT = {{BT}}
        BK_h = {{BK_h}}
        BV_h = {{BV_h}}
        num_stages_h = {{num_stages_h}}
        num_threads_h = {{num_threads_h}}
        BK_o = {{BK_o}}
        BV_o = {{BV_o}}
        num_stages_o = {{num_stages_o}}
        num_threads_o = {{num_threads_o}}

        # decay_mod here
        {{decay_mod_expr | indent(8)}}

        # k_mod here
        {{k_mod_expr | indent(8)}}

        # v_mod here
        {{v_mod_expr | indent(8)}}

        decay_cumsum = chunk_local_cumsum_scalar(
            {{decay_name}}, BT
        )
        chunk_fwd_h_mod = tl.profiler.cached(chunk_fwd_h, {{output_idx_list_h}}, BATCH, HQ,HK, H, N_CTX, D_HEAD, D_HEADV, BT, BK_h, BV_h, num_stages_h, num_threads_h)
        output_idx_list = {{output_idx_list_o}}# [5,]
        chunk_fwd_o_mod = tl.profiler.cached(chunk_o, output_idx_list, BATCH, HQ,HK, H, N_CTX, D_HEAD, D_HEADV, BT, BK_o, BV_o, num_stages_o, num_threads_o)

        h = chunk_fwd_h_mod({{k_name}}, {{v_name}}, decay_cumsum, {{custom_inputs_list_h}})
        o = chunk_fwd_o_mod(h, {{q_name}}, {{k_name}}, {{v_name}}, decay_cumsum, {{custom_inputs_list_o}})

        ctx.save_for_backward(q, k, v, decay, {{custom_inputs_list}} ) # , decay_cumsum
        ctx.BT = BT
        return o

    @staticmethod
    def backward(ctx, do):
        do = do.contiguous()
        BT = {{BT_BWD}}# ctx.BT
        q, k, v, decay, {{custom_inputs_list}} = ctx.saved_tensors #  decay_cumsum

        BATCH, HQ, N_CTX, D_HEAD = q.shape
        D_HEADV = v.shape[-1]
        HK = k.shape[1]
        H = v.shape[1]
        NT = N_CTX // BT
        
        BK_h = {{BK_h}}
        BV_h = {{BV_h}}
        num_stages_h = {{num_stages_h}}
        num_threads_h = {{num_threads_h}}
        BK_dh = {{BK_dh}}
        BV_dh = {{BV_dh}}
        num_stages_dh = {{num_stages_dh}}
        num_threads_dh = {{num_threads_dh}}
        BK_dqkg = {{BK_dqkg}}
        BV_dqkg = {{BV_dqkg}}
        num_stages_dqkg = {{num_stages_dqkg}}
        num_threads_dqkg = {{num_threads_dqkg}}
        BK_dv = {{BK_dv}}
        BV_dv = {{BV_dv}}
        num_stages_dv = {{num_stages_dv}}
        num_threads_dv = {{num_threads_dv}}
        
        # decay_mod here
        {{decay_mod_expr1 | indent(8)}}

        # k_mod here
        {{k_mod_expr1 | indent(8)}}

        # v_mod here
        {{v_mod_expr1 | indent(8)}}
        
        # q_mod here
        {{q_mod_expr1 | indent(8)}}
        
        decay_cumsum = chunk_local_cumsum_scalar(
            {{decay_name2}}, BT
        )
        
        chunk_fwd_h_mod = tl.profiler.cached(chunk_fwd_h, {{output_idx_list_h}}, BATCH, HQ,HK, H, N_CTX, D_HEAD, D_HEADV, BT, BK_h, BV_h, num_stages_h, num_threads_h)
        chunk_bwd_dh_mod = tl.profiler.cached(chunk_bwd_kernel_dh, [5,], BATCH, HQ, HK, H, N_CTX, D_HEAD, D_HEADV, BT, BK_dh, BV_dh, num_stages_dh, num_threads_dh)
        chunk_bwd_dqkg_mod = tl.profiler.cached(chunk_bwd_dqkg, [7,8,9,], BATCH, HQ, HK, H, N_CTX, D_HEAD, D_HEADV, BT, BK_dqkg, BV_dqkg, num_stages_dqkg, num_threads_dqkg)
        chunk_bwd_dv_mod = tl.profiler.cached(chunk_bwd_kernel_dv, [5,], BATCH, HQ, HK, H, N_CTX, D_HEAD, D_HEADV, BT, BK_dv, BV_dv, num_stages_dv, num_threads_dv)
        
        h = chunk_fwd_h_mod({{k_name2}}, {{v_name2}}, decay_cumsum, {{custom_inputs_list_h}})
        
        # k_mod 2 (2 not the same time as 1)
        {{k_mod_expr_2 | indent(8)}}
        # v_mod 2
        {{v_mod_expr_2 | indent(8)}}
        
        dh = chunk_bwd_dh_mod({{q_name1}}, {{k_name1}}, {{v_name1}}, decay_cumsum, do)
        dq, dk, ddecay = chunk_bwd_dqkg_mod({{q_name1}}, {{k_name1}}, {{v_name1}}, h, decay_cumsum, do, dh)
        if HQ < H:
            dq =  dq.view(BATCH, HQ, H//HQ, N_CTX, D_HEAD).sum(2)
        if HK < H:
            dk = dk.view(BATCH, HK, H//HK, N_CTX, D_HEAD).sum(2)
        ddecay = ddecay.sum(0)
        dg2 = torch.empty(ddecay.shape, dtype=torch.float32, device=ddecay.device)
        compute_final_dg[(NT, BATCH*H)](ddecay, dg2, T=N_CTX, BT=BT)
        dv = chunk_bwd_dv_mod({{q_name1}}, {{k_name1}}, decay_cumsum, do, dh)
        
        # v_bwd
        {{v_mod_bwd_expr | indent(8)}}
        # decay_mod_bwd
        {{decay_mod_bwd_expr | indent(8)}}
        # k_bwd
        {{k_mod_bwd_expr | indent(8)}}
        # q_bwd
        {{q_mod_bwd_expr | indent(8)}}
        
        return {{dq_name}}, {{dk_name}}, {{dv_name}}, {{ddecay_name}}, {{custom_inputs_grad_list}}
        
        

linear_attention = LinearAttention.apply

# --------------- TL_TUNNER
# AUTOtune
from autotuner.arch.H100 import H100

def generate_config(BATCH, HQ, HK, H, N_CTX, D_HEAD, D_HEADV, BT,device=H100()):
    # BTs = [32,64,128,192,256]
    BK_hs = [32,64,128,192,256]
    BV_hs = [32,64,128,192,256]
    num_stages_hs = [1,2,3,4]
    num_threads_hs = [128,256]
    BK_os = [32,64,128,256]
    BV_os = [32,64,128,256]
    num_stages_os = [1,2,3,4]
    num_threads_os = [128,256]
    
    # H100
    MMA_ATOM_M = device.mma_primitive[0]# 64
    MMA_ATOM_TRHEADS = device.threads_per_mma # 128
    smem_cap = device.smem_cap # 232448
    reg_cap = device.reg_cap * 4 # 65536 * 4
    reg_cap_per_thread = device.register_per_thread * 4 # 255 * 4
    
    # input shape
    # BTs = [bt for bt in BTs if N_CTX % bt == 0]
    BK_hs = [bk for bk in BK_hs if D_HEAD % bk == 0]
    BV_hs = [bv for bv in BV_hs if D_HEADV % bv == 0]
    BK_os = [bk for bk in BK_os if D_HEAD % bk == 0]
    BV_os = [bv for bv in BV_os if D_HEADV % bv == 0]
       
    config_h = []
    for BK_h, BV_h, num_stages_h, num_threads_h in itertools.product(BK_hs, BV_hs, num_stages_hs, num_threads_hs):
        sharedmem_chunk_h = 2 * (
            BK_h * BV_h + (BT * BK_h + BT * BV_h + BT ) 
            * num_stages_h)
        reg_chunk_h = 4 * (
            BK_h * BV_h
        )
        conditions_h = [
            num_stages_h > N_CTX // BT,
            BK_h % (MMA_ATOM_M) != 0,
            sharedmem_chunk_h > smem_cap,
            reg_chunk_h > reg_cap and reg_chunk_h > reg_cap_per_thread * num_threads_h,
        ]
        if any(conditions_h):
            continue
        config_h.append( {
        # "BT": BT,
        "BK_h": BK_h,
        "BV_h": BV_h,
        "num_stages_h": num_stages_h,
        "num_threads_h": num_threads_h,
        })
    
    config_o = []
    for BK_o, BV_o, num_stages_o, num_threads_o in itertools.product(BK_os, BV_os, num_stages_os, num_threads_os):
        sharedmem_chunk_o = 2 * (
            BT * BK_o * num_stages_o + BT * BK_o * num_stages_o + BK_o * BV_o* num_stages_o +
            BT * BV_o 
        )
        reg_chunk_o = 4 * (
            BT * BT + BT * BV_o
        )
        conditions_o = [
            num_stages_o > D_HEAD // BK_o,
            BT % (MMA_ATOM_M*(num_threads_o//MMA_ATOM_TRHEADS)) != 0,
            sharedmem_chunk_o > smem_cap,
            reg_chunk_o > reg_cap and reg_chunk_o > reg_cap_per_thread * num_threads_o,
        ]
        if any(conditions_o):
            continue
        config_o.append( {
        # "BT": BT,
        "BK_o": BK_o,
        "BV_o": BV_o,
        "num_stages_o": num_stages_o,
        "num_threads_o": num_threads_o,
        })
        
    return config_h, config_o

def generate_config_bwd(BATCH, HQ, HK, H, N_CTX, D_HEAD, D_HEADV, BT,device=H100()):
    # BTs = [32,64,128,192,256]
    BK_hs = [32,64,128,192,256]
    BV_hs = [32,64,128,192,256]
    num_stages_hs = [1,2,3,4]
    num_threads_hs = [128,256]
    BK_dhs = [32,64,128,192,256]
    BV_dhs = [32,64,128,192,256]
    num_stages_dhs = [1,2,3,4]
    num_threads_dhs = [128,256]
    BK_dqkgs = [32,64,128,256]
    BV_dqkgs = [32,64,128,256]
    num_stages_dqkgs = [1,2,3,4]
    num_threads_dqkgs = [128,256]
    BK_dvs = [32,64,128,256]
    BV_dvs = [32,64,128,256]
    num_stages_dvs = [1,2,3,4]
    num_threads_dvs = [128,256]
    
    # H100
    MMA_ATOM_M = device.mma_primitive[0]# 64
    MMA_ATOM_TRHEADS = device.threads_per_mma # 128
    smem_cap = device.smem_cap # 232448
    reg_cap = device.reg_cap * 4 # 65536 * 4
    reg_cap_per_thread = device.register_per_thread * 4 # 255 * 4
    
    # input shape
    # BTs = [bt for bt in BTs if N_CTX % bt == 0]
    BK_hs = [bk for bk in BK_hs if D_HEAD % bk == 0]
    BV_hs = [bv for bv in BV_hs if D_HEADV % bv == 0]
    BK_dhs = [bk for bk in BK_dhs if D_HEAD % bk == 0]
    BV_dhs = [bv for bv in BV_dhs if D_HEADV % bv == 0]
    BK_dqkgs = [bk for bk in BK_dqkgs if D_HEAD % bk == 0]
    BV_dqkgs = [bv for bv in BV_dqkgs if D_HEADV % bv == 0]
    BK_dvs = [bk for bk in BK_dvs if D_HEAD % bk == 0]
    BV_dvs = [bv for bv in BV_dvs if D_HEADV % bv == 0]
    
    config_h = []
    for BK_h, BV_h, num_stages_h, num_threads_h in itertools.product(BK_hs, BV_hs, num_stages_hs, num_threads_hs):
        sharedmem_chunk_h = 2 * (
            BK_h * BV_h + (BT * BK_h + BT * BV_h + BT ) 
            * num_stages_h)
        reg_chunk_h = 4 * (
            BK_h * BV_h
        )
        conditions_h = [
            num_stages_h > N_CTX // BT,
            BK_h % (MMA_ATOM_M) != 0,
            sharedmem_chunk_h > smem_cap,
            reg_chunk_h > reg_cap and reg_chunk_h > reg_cap_per_thread * num_threads_h,
        ]
        if any(conditions_h):
            continue
        config_h.append( {
        # "BT": BT,
        "BK_h": BK_h,
        "BV_h": BV_h,
        "num_stages_h": num_stages_h,
        "num_threads_h": num_threads_h,
        })
    
    config_dh = []
    for BK_dh, BV_dh, num_stages_dh, num_threads_dh in itertools.product(BK_dhs, BV_dhs, num_stages_dhs, num_threads_dhs):
        # sharedmem_chunk_dh = 2 * (
        #     BK_dh * BV_dh + (BT * BK_dh + BT * BV_dh + BT ) 
        #     * num_stages_dh)
        # reg_chunk_dh = 4 * (
        #     BK_dh * BV_dh
        # )
        conditions_dh = [
            num_stages_dh > N_CTX // BT,
            BK_dh % (MMA_ATOM_M) != 0,
            # sharedmem_chunk_dh > smem_cap,
            # reg_chunk_dh > reg_cap and reg_chunk_dh > reg_cap_per_thread * num_threads_dh,
        ]
        if any(conditions_dh):
            continue
        config_dh.append( {
        # "BT": BT,
        "BK_dh": BK_dh,
        "BV_dh": BV_dh,
        "num_stages_dh": num_stages_dh,
        "num_threads_dh": num_threads_dh,
        })
        
    config_dqkg = []
    for BK_dqkg, BV_dqkg, num_stages_dqkg, num_threads_dqkg in itertools.product(BK_dqkgs, BV_dqkgs, num_stages_dqkgs, num_threads_dqkgs):
        # sharedmem_chunk_dqkg = 2 * (
        #     BT * BK_dqkg * num_stages_dqkg + BT * BK_dqkg * num_stages_dqkg + BK_dqkg * BV_dqkg* num_stages_dqkg +
        #     BT * BV_dqkg 
        # )
        # reg_chunk_dqkg = 4 * (
        #     BT * BT + BT * BV_dqkg
        # )
        conditions_dqkg = [
            num_stages_dqkg > D_HEADV // BV_dqkg,
            BT % (MMA_ATOM_M*(num_threads_dqkg//MMA_ATOM_TRHEADS)) != 0,
        ]
        if any(conditions_dqkg):
            continue
        config_dqkg.append( {
        # "BT": BT,
        "BK_dqkg": BK_dqkg,
        "BV_dqkg": BV_dqkg,
        "num_stages_dqkg": num_stages_dqkg,
        "num_threads_dqkg": num_threads_dqkg,
        })
    
    config_dv = []
    for BK_dv, BV_dv, num_stages_dv, num_threads_dv in itertools.product(BK_dvs, BV_dvs, num_stages_dvs, num_threads_dvs):
        # sharedmem_chunk_dv = 2 * (
        #     BT * BK_dv * num_stages_dv + BT * BK_dv * num_stages_dv + BK_dv * BV_dv* num_stages_dv +
        #     BT * BV_dv 
        # )
        # reg_chunk_dv = 4 * (
        #     BT * BT + BT * BV_dv
        # )
        conditions_dv = [
            num_stages_dv > D_HEAD // BK_dv,
            BT % (MMA_ATOM_M*(num_threads_dv//MMA_ATOM_TRHEADS)) != 0,
            # sharedmem_chunk_dv > smem_cap,
            # reg_chunk_dv > reg_cap and reg_chunk_dv > reg_cap_per_thread * num_threads_dv,
        ]
        if any(conditions_dv):
            continue
        config_dv.append( {
        # "BT": BT,
        "BK_dv": BK_dv,
        "BV_dv": BV_dv,
        "num_stages_dv": num_stages_dv,
        "num_threads_dv": num_threads_dv,
        })
    return config_h, config_dh, config_dqkg, config_dv

from autotuner.attnfwd_tunner_engine2 import tl_tune

def autotune(B, HQ, HK, H, Tlen, D, DV, file_path="mamba2",device=H100()):
            
    BTs = [32,64,128,192,256]
 
    best_config = {}
    best_latency = 1e6
    for BT in BTs:
        config_h,config_o = generate_config(B, HQ, HK, H, Tlen, D, DV, BT, device)
        if len(config_h) == 0 or len(config_o) == 0:
            continue
        problem_keys = {
            "B": B, "HQ": HQ, "HK": HK, "H": H, "Tlen": Tlen, "D": D, "DV": DV, "BT": BT
        }
        best_config_h, best_latency_h = tl_tune(chunk_fwd_h, problem_keys, config_h, {{output_idx_list_h}}, file_path=f"{file_path}_h.json")
        best_config_o, best_latency_o  = tl_tune(chunk_o, problem_keys, config_o, {{output_idx_list_o}}, file_path=f"{file_path}_o.json")
        if best_latency_h + best_latency_o < best_latency:
            best_latency = best_latency_h + best_latency_o
            best_config = {
                "BT": BT,
                "BK_h": best_config_h["BK_h"],
                "BV_h": best_config_h["BV_h"],
                "num_stages_h": best_config_h["num_stages_h"],
                "num_threads_h": best_config_h["num_threads_h"],
                "BK_o": best_config_o["BK_o"],
                "BV_o": best_config_o["BV_o"],
                "num_stages_o": best_config_o["num_stages_o"],
                "num_threads_o": best_config_o["num_threads_o"],
            }
    
    return best_config, best_latency
   
def autotune_bwd(B, HQ, HK, H, Tlen, D, DV, file_path="mamba2",device=H100()):
            
    BTs = [32,64,128,192,256]
 
    best_config = {}
    best_latency = 1e6
    for BT in BTs:
        config_h, config_dh,config_dqkg, config_dv = generate_config_bwd(B, HQ, HK, H, Tlen, D, DV, BT, device)
        if len(config_h) == 0 or len(config_dh) == 0 or len(config_dqkg) == 0 or len(config_dv) == 0:
            continue
        problem_keys = {
            "B": B, "HQ": HQ, "HK": HK, "H": H, "Tlen": Tlen, "D": D, "DV": DV, "BT": BT
        }
        best_config_h, best_latency_h = tl_tune(chunk_fwd_h, problem_keys, config_h, {{output_idx_list_h}}, file_path=f"{file_path}_h.json")
        best_config_dh, best_latency_dh = tl_tune(chunk_bwd_kernel_dh, problem_keys, config_dh, [5,], file_path=f"{file_path}_dh.json")
        best_config_dqkg, best_latency_dqkg = tl_tune(chunk_bwd_dqkg, problem_keys, config_dqkg, [7,8,9,], file_path=f"{file_path}_dqkg.json")
        best_config_dv, best_latency_dv = tl_tune(chunk_bwd_kernel_dv, problem_keys, config_dv, [5,], file_path=f"{file_path}_dv.json")
        # print(BT, best_latency_h, best_latency_dh, best_latency_dqkg, best_latency_dv)
        if best_latency_h + best_latency_dh + best_latency_dqkg + best_latency_dv < best_latency:
            best_latency = best_latency_h + best_latency_dh + best_latency_dqkg + best_latency_dv
            best_config = {
                "BT_BWD": BT,
                "BK_h": best_config_h["BK_h"],
                "BV_h": best_config_h["BV_h"],
                "num_stages_h": best_config_h["num_stages_h"],
                "num_threads_h": best_config_h["num_threads_h"],
                "BK_dh": best_config_dh["BK_dh"],
                "BV_dh": best_config_dh["BV_dh"],
                "num_stages_dh": best_config_dh["num_stages_dh"],
                "num_threads_dh": best_config_dh["num_threads_dh"],
                "BK_dqkg": best_config_dqkg["BK_dqkg"],
                "BV_dqkg": best_config_dqkg["BV_dqkg"],
                "num_stages_dqkg": best_config_dqkg["num_stages_dqkg"],
                "num_threads_dqkg": best_config_dqkg["num_threads_dqkg"],
                "BK_dv": best_config_dv["BK_dv"],
                "BV_dv": best_config_dv["BV_dv"],
                "num_stages_dv": best_config_dv["num_stages_dv"],
                "num_threads_dv": best_config_dv["num_threads_dv"],
            }
    
    return best_config, best_latency

