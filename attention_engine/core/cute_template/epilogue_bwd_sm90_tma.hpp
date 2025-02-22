/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cutlass/cutlass.h>
#include "cute/tensor.hpp"

#include "cutlass/gemm/collective/collective_builder.hpp"

#include "named_barrier.hpp"
#include "utils.h"

namespace flash {

using namespace cute;

template <class TileShape_MNK_, class TileShape_MNK_V_,class Element_, int NumEpilogueThreads_, bool Varlen_>
struct CollectiveEpilogueBwd {

    using TileShape_MNK = TileShape_MNK_;
    using TileShape_MNK_V = TileShape_MNK_V_;
    using Element = Element_;
    static constexpr int NumEpilogueThreads = NumEpilogueThreads_;
    static constexpr bool Varlen = Varlen_;

    using GmemTiledCopydKVTMA = cute::SM90_TMA_STORE;

    // These are for storing the output tensor without TMA (e.g., for setting output to zero)
    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(get<2>(TileShape_MNK{}) % kGmemElemsPerLoad == 0, "QKHeaddim must be a multiple of kGmemElemsPerLoad");
    static_assert(get<2>(TileShape_MNK_V{}) % kGmemElemsPerLoad == 0, "VHeaddim must be a multiple of kGmemElemsPerLoad");
    static constexpr int kQKHeadDim = get<2>(TileShape_MNK{});
    static constexpr int kVHeadDim = get<2>(TileShape_MNK_V{});
    static constexpr int kGmemThreadsPerRow = cutlass::gcd(kQKHeadDim / kGmemElemsPerLoad, NumEpilogueThreads); // TODO:check kVHeadDim
    static_assert(NumEpilogueThreads % kGmemThreadsPerRow == 0, "NumEpilogueThreads must be a multiple of kGmemThreadsPerRow");
    using GmemLayoutAtom = Layout<Shape <Int<NumEpilogueThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                  Stride<Int<kGmemThreadsPerRow>, _1>>;
    using GmemTiledCopydKV = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));  // Val layout, 8 or 16 vals per store

    using SmemLayoutAtomdKTMA = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<1>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
    using SmemLayoutAtomdVTMA = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<1>(TileShape_MNK_V{})), decltype(cute::get<2>(TileShape_MNK_V{}))>());
    using SmemLayoutdKTMA = decltype(tile_to_shape(SmemLayoutAtomdKTMA{}, select<1, 2>(TileShape_MNK{})));
    using SmemLayoutdVTMA = decltype(tile_to_shape(SmemLayoutAtomdVTMA{}, select<1, 2>(TileShape_MNK_V{})));

    // If we don't use TMA
    static constexpr int kBlockKSmem = kQKHeadDim % 64 == 0 ? 64 : (kQKHeadDim % 32 == 0 ? 32 : 16); // TODO: check kVHeadDim
    static constexpr int kSwizzle = kBlockKSmem == 64 ? 3 : (kBlockKSmem == 32 ? 2 : 1);
    using SmemLayoutAtomdKVSTG =
        decltype(composition(Swizzle<kSwizzle, 3, 3>{},
                             Layout<Shape<Int<8>, Int<kBlockKSmem>>,
                             Stride<Int<kBlockKSmem>, _1>>{}));

    using SmemLayoutAtomdK = std::conditional_t<!Varlen, SmemLayoutAtomdKTMA, SmemLayoutAtomdKVSTG>;
    using SmemLayoutAtomdV = std::conditional_t<!Varlen, SmemLayoutAtomdVTMA, SmemLayoutAtomdKVSTG>;
    using SmemLayoutdK = decltype(tile_to_shape(SmemLayoutAtomdK{}, select<1, 2>(TileShape_MNK{})));
    using SmemLayoutdV = decltype(tile_to_shape(SmemLayoutAtomdV{}, select<1, 2>(TileShape_MNK_V{})));

    using SmemCopyAtomdKV = Copy_Atom<cute::SM90_U32x4_STSM_N, Element>;

    struct TensorStorage : cute::aligned_struct<128> {
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutdK>> smem_dk;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutdV>> smem_dv;
    };

    using ShapedKV = cute::Shape<int32_t, int32_t, int32_t, int32_t>;  // (seqlen_q, d, head, batch)
    using StridedKV = cute::Stride<int64_t, _1, int64_t, int64_t>;
    using LayoutdKV = cute::Layout<ShapedKV, StridedKV>;

    using TMA_dK = decltype(make_tma_copy(
        GmemTiledCopydKVTMA{},
        make_tensor(make_gmem_ptr(static_cast<Element*>(nullptr)), ShapedKV{}, StridedKV{}),
        SmemLayoutdKTMA{},
        select<1, 2>(TileShape_MNK{}),
        _1{}));  // no mcast for dKV
    using TMA_dV = decltype(make_tma_copy(
        GmemTiledCopydKVTMA{},
        make_tensor(make_gmem_ptr(static_cast<Element*>(nullptr)), ShapedKV{}, StridedKV{}),
        SmemLayoutdVTMA{},
        select<1, 2>(TileShape_MNK_V{}),
        _1{}));  // no mcast for dKV
    // Host side kernel arguments
    struct Arguments {
        Element* ptr_dK;
        ShapedKV const shape_dK;
        StridedKV const stride_dK;
        Element* ptr_dV;
        ShapedKV const shape_dV;
        StridedKV const stride_dV;
        int const* cu_seqlens = nullptr;
    };

    // Device side kernel params
    struct Params {
        Element* ptr_dK;
        ShapedKV const shape_dK;
        StridedKV const stride_dK;
        Element* ptr_dV;
        ShapedKV const shape_dV;
        StridedKV const stride_dV;
        TMA_dK tma_store_dK;
        TMA_dV tma_store_dV;
        int const* cu_seqlens = nullptr;
    };

    static Params
    to_underlying_arguments(Arguments const& args) {
        if constexpr (Varlen) {
            assert (args.cu_seqlens != nullptr);
        }
        Tensor mdK = make_tensor(make_gmem_ptr(args.ptr_dK), args.shape_dK, args.stride_dK);
        Tensor mdV = make_tensor(make_gmem_ptr(args.ptr_dV), args.shape_dV, args.stride_dV);
        TMA_dK tma_store_dK = make_tma_copy(
            GmemTiledCopydKVTMA{},
            mdK,
            SmemLayoutdKTMA{},
            select<1, 2>(TileShape_MNK{}),
            _1{}); // no mcast for dKV
        TMA_dV tma_store_dV = make_tma_copy(
            GmemTiledCopydKVTMA{},
            mdV,
            SmemLayoutdVTMA{},
            select<1, 2>(TileShape_MNK_V{}),
            _1{}); // no mcast for dKV
        return {args.ptr_dK, args.shape_dK, args.stride_dK, args.ptr_dV, args.shape_dV, args.stride_dV,
                tma_store_dK, tma_store_dV, args.cu_seqlens};
    }

    /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
    CUTLASS_DEVICE
    static void prefetch_tma_descriptors(Params const& params) {
        if constexpr (!Varlen) {
            cute::prefetch_tma_descriptor(params.tma_store_dK.get_tma_descriptor());
            cute::prefetch_tma_descriptor(params.tma_store_dV.get_tma_descriptor());
        }
    }

    template <typename SharedStorage, typename FrgTensordK, typename FrgTensordV, typename TiledMmadK, typename TiledMmadV>
    CUTLASS_DEVICE void
    store(Params const& params,
          FrgTensordK const& tdKrdK,
          FrgTensordV const& tdVrdV,
          SharedStorage& shared_storage,
          TiledMmadK tiled_mmadK,
          TiledMmadV tiled_mmadV,
          int thread_idx,
          cute::tuple<int32_t, int32_t, int32_t> const& block_coord
          ) {

        auto [n_block, bidh, bidb] = block_coord;
        Tensor sdK = make_tensor(make_smem_ptr(shared_storage.epilogue.smem_dk.data()), SmemLayoutdK{});
        Tensor sdV = make_tensor(make_smem_ptr(shared_storage.epilogue.smem_dv.data()), SmemLayoutdV{});
        auto smem_tiled_copy_dK = make_tiled_copy_C(SmemCopyAtomdKV{}, tiled_mmadK);
        auto smem_tiled_copy_dV = make_tiled_copy_C(SmemCopyAtomdKV{}, tiled_mmadV);
        auto smem_thr_copy_dK = smem_tiled_copy_dK.get_thread_slice(thread_idx);
        auto smem_thr_copy_dV = smem_tiled_copy_dV.get_thread_slice(thread_idx);
        // if (blockIdx.x == 0 && threadIdx.x == 128) { 
        //     printf("tdVrdV#########################: \n");
        //     print_tensor(tdVrdV); }
        Tensor tdVrdV_out = flash::convert_type<Element>(tdVrdV);
        // if (blockIdx.x == 0 && threadIdx.x == 128) { 
        //     printf("tdVrdV_out#########################: \n");
        //     print_tensor(tdVrdV_out); }
        Tensor tdKrdK_out = flash::convert_type<Element>(tdKrdK);
        Tensor taccdKrdK = smem_thr_copy_dK.retile_S(tdKrdK_out);        // ((Atom,AtomNum), MMA_M, MMA_N)
        Tensor taccdVrdV = smem_thr_copy_dV.retile_S(tdVrdV_out);        // ((Atom,AtomNum), MMA_M, MMA_N)
        Tensor taccdKsdK = smem_thr_copy_dK.partition_D(sdK);     // ((Atom,AtomNum),PIPE_M,PIPE_N)
        Tensor taccdVsdV = smem_thr_copy_dV.partition_D(sdV);     // ((Atom,AtomNum),PIPE_M,PIPE_N)

        // Make sure all WGs have finished reading K and V

        cutlass::arch::NamedBarrier::sync(NumEpilogueThreads, static_cast<int>(BwdNamedBarriers::KVEmpty) /*id*/);
        cute::copy(smem_tiled_copy_dV, taccdVrdV, taccdVsdV);
        cute::copy(smem_tiled_copy_dK, taccdKrdK, taccdKsdK);
        if constexpr (!Varlen) {
            cutlass::arch::fence_view_async_shared(); // ensure smem writes are visible to TMA
            cutlass::arch::NamedBarrier::arrive(NumEpilogueThreads + cutlass::NumThreadsPerWarp,
                                                cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);

            Tensor mdK = params.tma_store_dK.get_tma_tensor(params.shape_dK);
            Tensor mdV = params.tma_store_dV.get_tma_tensor(params.shape_dV);
            Tensor gdK = local_tile(mdK(_, _, bidh, bidb), select<1, 2>(TileShape_MNK{}), make_coord(n_block, _0{}));  // (M, K)
            Tensor gdV = local_tile(mdV(_, _, bidh, bidb), select<1, 2>(TileShape_MNK_V{}), make_coord(n_block, _0{}));  // (M, K)
            auto block_tma_dK = params.tma_store_dK.get_slice(_0{});
            auto block_tma_dV = params.tma_store_dV.get_slice(_0{});
            Tensor tdKgdK = block_tma_dK.partition_D(gdK);  // (TMA, TMA_M, TMA_K)
            Tensor tdKsdK = block_tma_dK.partition_S(sdK); // (TMA, TMA_M, TMA_K)
            Tensor tdVgdV = block_tma_dV.partition_D(gdV);  // (TMA, TMA_M, TMA_K)
            Tensor tdVsdV = block_tma_dV.partition_S(sdV); // (TMA, TMA_M, TMA_K)
            int warp_idx_sync = __shfl_sync(0xffffffff, thread_idx / cutlass::NumThreadsPerWarp, 0);
            if (warp_idx_sync == NumEpilogueThreads / cutlass::NumThreadsPerWarp - 1) {
                cutlass::arch::NamedBarrier::sync(NumEpilogueThreads + cutlass::NumThreadsPerWarp,
                                                cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
                int const lane_predicate = cute::elect_one_sync();
                if (lane_predicate) {
                    cute::copy(params.tma_store_dV, tdVsdV, tdVgdV);
                    cute::copy(params.tma_store_dK, tdKsdK, tdKgdK);
                    tma_store_arrive();
                }
            }

        } else {
            cutlass::arch::NamedBarrier::sync(NumEpilogueThreads, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
            bool const is_varlen = params.cu_seqlens != nullptr;
            int const offset = !is_varlen ? 0 : params.cu_seqlens[bidb];
            int const seqlen = !is_varlen ? get<0>(params.shape_dK) : params.cu_seqlens[bidb + 1] - params.cu_seqlens[bidb];

            Tensor mdK = make_tensor(make_gmem_ptr(params.ptr_dK), params.shape_dK, params.stride_dK)(_, _, bidh, !is_varlen ? bidb : 0);
            Tensor gdK = local_tile(cute::domain_offset(make_coord(offset, _0{}), mdK), select<1, 2>(TileShape_MNK{}), make_coord(n_block, _0{}));  // (M, K)
            Tensor mdV = make_tensor(make_gmem_ptr(params.ptr_dV), params.shape_dV, params.stride_dV)(_, _, bidh, !is_varlen ? bidb : 0);
            Tensor gdV = local_tile(cute::domain_offset(make_coord(offset, _0{}), mdV), select<1, 2>(TileShape_MNK_V{}), make_coord(n_block, _0{}));  // (M, K)

            GmemTiledCopydKV gmem_tiled_copy_dKV;
            auto gmem_thr_copy_dKV = gmem_tiled_copy_dKV.get_thread_slice(thread_idx);
            Tensor tdKVgdV = gmem_thr_copy_dKV.partition_D(gdV);
            Tensor tdKVsdV = gmem_thr_copy_dKV.partition_S(sdV); // (TMA, TMA_M, TMA_K)
            Tensor tdKVgdK = gmem_thr_copy_dKV.partition_D(gdK);
            Tensor tdKVsdK = gmem_thr_copy_dKV.partition_S(sdK); // (TMA, TMA_M, TMA_K)
            Tensor tdKVrdV = make_fragment_like(tdKVgdV);
            Tensor tdKVrdK = make_fragment_like(tdKVgdK);
            cute::copy(gmem_tiled_copy_dKV, tdKVsdV, tdKVrdV);
            cute::copy(gmem_tiled_copy_dKV, tdKVsdK, tdKVrdK);
            // Construct identity layout for gdKV
            Tensor cdK = cute::make_identity_tensor(select<1, 2>(TileShape_MNK{}));  // (BLK_M,BLK_K) -> (blk_m,blk_k)
            Tensor cdV = cute::make_identity_tensor(select<1, 2>(TileShape_MNK_V{}));  // (BLK_M,BLK_K) -> (blk_m,blk_k)
            // Repeat the partitioning with identity layouts
            Tensor tdKcdK = gmem_thr_copy_dKV.partition_D(cdK);
            Tensor tdVcdV = gmem_thr_copy_dKV.partition_D(cdV);
            Tensor tdKVpdV = make_tensor<bool>(make_shape(size<2>(tdKVgdV)));
            Tensor tdKVpdK = make_tensor<bool>(make_shape(size<2>(tdKVgdK)));
            #pragma unroll
            for (int k = 0; k < size(tdKVpdK); ++k) { tdKVpdK(k) = get<1>(tdKcdK(_0{}, _0{}, k)) < get<1>(params.shape_dK); }
            #pragma unroll
            for (int k = 0; k < size(tdKVpdV); ++k) { tdKVpdV(k) = get<1>(tdVcdV(_0{}, _0{}, k)) < get<1>(params.shape_dV); }
            static constexpr int kBlockN = get<1>(TileShape_MNK{});
            // Clear_OOB_K must be false since we don't want to write zeros to gmem
            flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
                gmem_tiled_copy_dKV, tdKVrdV, tdKVgdV, tdVcdV, tdKVpdV, seqlen - n_block * kBlockN
            );
            flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
                gmem_tiled_copy_dKV, tdKVrdK, tdKVgdK, tdKcdK, tdKVpdK, seqlen - n_block * kBlockN
            );
        }
    }

    CUTLASS_DEVICE void
    store_tail() {
        if constexpr (!Varlen) { tma_store_wait<0>(); }
    }

    // Write 0 to dK and dV
    CUTLASS_DEVICE void
    store_zero(
         Params const& params,
         int thread_idx,
         cute::tuple<int32_t, int32_t, int32_t> const& block_coord
         ) {
        static constexpr int kBlockN = get<1>(TileShape_MNK{});
        auto [n_block, bidh, bidb] = block_coord;
        bool const is_varlen = Varlen && params.cu_seqlens != nullptr;
        int const offset = !is_varlen ? 0 : params.cu_seqlens[bidb];
        int const seqlen = !is_varlen ? get<0>(params.shape_dK) : params.cu_seqlens[bidb + 1] - offset;

        Tensor mdK = make_tensor(make_gmem_ptr(params.ptr_dK), params.shape_dK, params.stride_dK)(_, _, bidh, !is_varlen ? bidb : 0);
        Tensor gdK = local_tile(cute::domain_offset(make_coord(offset, _0{}), mdK), select<1, 2>(TileShape_MNK{}), make_coord(n_block, _0{}));  // (M, K)
        Tensor mdV = make_tensor(make_gmem_ptr(params.ptr_dV), params.shape_dV, params.stride_dV)(_, _, bidh, !is_varlen ? bidb : 0);
        Tensor gdV = local_tile(cute::domain_offset(make_coord(offset, _0{}), mdV), select<1, 2>(TileShape_MNK_V{}), make_coord(n_block, _0{}));  // (M, K)

        GmemTiledCopydKV gmem_tiled_copy_dKV;
        auto gmem_thr_copy_dKV = gmem_tiled_copy_dKV.get_thread_slice(thread_idx);
        Tensor tdKVgdK = gmem_thr_copy_dKV.partition_D(gdK);
        Tensor tdKVgdV = gmem_thr_copy_dKV.partition_D(gdV);
        Tensor tdKVrdK = make_fragment_like(tdKVgdK);
        clear(tdKVrdK);
        Tensor tdKVrdV = make_fragment_like(tdKVgdV);
        clear(tdKVrdV);
        // Construct identity layout for gdKV
        Tensor cdK = cute::make_identity_tensor(select<1, 2>(TileShape_MNK{}));  // (BLK_M,BLK_K) -> (blk_m,blk_k)
        Tensor cdV = cute::make_identity_tensor(select<1, 2>(TileShape_MNK_V{}));  // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tdKVcdK = gmem_thr_copy_dKV.partition_D(cdK);
        Tensor tdKVcdV = gmem_thr_copy_dKV.partition_D(cdV);
        Tensor tdKVpdK = make_tensor<bool>(make_shape(size<2>(tdKVgdK)));
        Tensor tdKVpdV = make_tensor<bool>(make_shape(size<2>(tdKVgdV)));
        #pragma unroll
        for (int k = 0; k < size(tdKVpdK); ++k) { tdKVpdK(k) = get<1>(tdKVcdK(_0{}, _0{}, k)) < get<1>(params.shape_dK); }
        #pragma unroll
        for (int k = 0; k < size(tdKVpdV); ++k) { tdKVpdV(k) = get<1>(tdKVcdV(_0{}, _0{}, k)) < get<1>(params.shape_dV); }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_dKV, tdKVrdK, tdKVgdK, tdKVcdK, tdKVpdK, seqlen - n_block * kBlockN
        );
        flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_dKV, tdKVrdV, tdKVgdV, tdKVcdV, tdKVpdV, seqlen - n_block * kBlockN
        );
    }

};

} // namespace flash
