import tvm
from typing import Dict, List, Tuple
import numpy as np

from ..arch import Arch
from ..config import Config, Stride, TileDict
from ..node import PrimFuncNode
from .common import factorize, get_all_factors
from .default import DefaultPolicy
from ..rasterization import *

class TensorCorePolicy(DefaultPolicy):
    def __init__(self, func: tvm.tir.PrimFunc, arch: Arch, tags:Dict={}) -> None:
        super().__init__(func, arch, tags)
        self.wmma_k = 16
        self.pipeline_stage:int = 1
        self.use_async_copy:bool = False
        self._legalize_info()
    
    def _legalize_info(self):
        pipleline_stage = self.prim_func_node.get_tag("pipeline_stage")
        if pipleline_stage:
            self.pipeline_stage = pipleline_stage
        else:
            if self.arch.compute_capability == "sm_80":
                self.pipeline_stage = 2
            else:
                self.pipeline_stage = 1
        use_async_copy = self.prim_func_node.get_tag("use_async_copy")
        if use_async_copy:
            self.use_async_copy = use_async_copy
        else:
            if self.arch.compute_capability == "sm_80":
                self.use_async_copy = 2
            else:
                self.use_async_copy = 1
            
    def _compute_tc_strides(self, node: PrimFuncNode, tile: List[int], rstep: Dict[str, int]={}) -> Tuple[Stride, Stride, Stride]:
        '''
            strides was used for shared memory padding, which is necessary for avoiding shared memory load bank conflict when we do not applying tensorcore layout.
        '''
        shapes = node.propogate_reduction_inputs(tile, rstep)
        AS_shape, BS_shape = shapes.values()
        CS_shape = tile
        A_ax_m, A_ax_k, B_ax_k, B_ax_n, C_ax_m, C_ax_n = node.infer_tensorcore_axis()
        # applying strides
        # TODO(leiwang1999): offset should be dynamically set. we can use tag -> enable_offset to control this option..
        offset = 8
        A_high_ax = min(A_ax_m, A_ax_k)
        B_high_ax = min(B_ax_n, B_ax_k)
        C_high_ax = min(C_ax_m, C_ax_n)
        A_stride = Stride(stride=np.prod(AS_shape[A_high_ax+1:]) + offset, ax=A_high_ax)
        B_stride = Stride(stride=np.prod(BS_shape[B_high_ax+1:]) + offset, ax=B_high_ax)
        C_stride = Stride(stride=np.prod(CS_shape[C_high_ax+1:]) + offset, ax=C_high_ax)
        return A_stride, B_stride, C_stride


    def infer_node_smem_usage(self, td: TileDict, node: PrimFuncNode):
        value, cached_tensors = super().infer_node_smem_usage(td, node)
        value *= self.pipeline_stage
        return value, cached_tensors

    def _assign_reduce_step(self, node):
        if not node.get_tag("tensorcore_config"):
            return super()._assign_reduce_step(node)
        result = {}
        for iter_info in node.raxis:
            iter_name = iter_info.var.name
            iter_dom = iter_info.dom
            if iter_dom % 16 > 0:
                result[iter_name] = 16 if iter_dom < 32 else 32 # padding case
            elif iter_dom % 32 == 0:
                result[iter_name] = 32
            else:
                return super()._assign_reduce_step(node)
        return result

    def _expand_reduce_axis(self, td):
        return

    def get_node_reduce_step_candidates(self, node):
        if not node.get_tag("tensorcore_config"):
            return super().get_node_reduce_step_candidates(node)
        else:
            # must be a a multiple of wmma_k
            return {k : [x * self.wmma_k for x in get_all_factors(node.raxis[k] // self.wmma_k)] for k in node.raxis}

    def check_tile_shape_isvalid(self, td: TileDict):
        for node in self.ordered_nodes:
            if node.get_tag("tensorcore_config"):
                ax_m, ax_n = node.get_tag("tensorcore_config")
                block_m, block_n = td.tile_map[node][ax_m], td.tile_map[node][ax_n]
                wmma_invalid = [block_m % wmma_m or block_n % wmma_n for wmma_m, wmma_n in [(16, 16), (8, 32), (32, 8)]]
                if all(wmma_invalid):
                    return False
                if any([y % x for x, y in zip(td.tile_map[node], node.get_space_dim())]):
                    return False
        return super().check_tile_shape_isvalid(td)

    def compute_node_stride_map(self, node: PrimFuncNode, td: TileDict):
        if not node.get_tag("tensorcore_config"):
            return super().compute_node_stride_map(node, td)
        AS_stride, BS_stride, C_stride = self._compute_tc_strides(node, td.get_tile(node), td.get_rstep(node))
        A_stride, B_stride, _ = self._compute_tc_strides(node, td.get_tile(node))
        output_strides = {
            int(i + len(node.input_buffers)): Stride() for i, _ in enumerate(node.output_buffers)
        }
        tensor_strides = {}

        return output_strides, tensor_strides

    def _assign_block_size(self, node: PrimFuncNode, td: TileDict, block_size: int):
        if not node.get_tag("tensorcore_config"):
            return super()._assign_block_size(node, td, block_size)
        ax_m, ax_n = node.get_tag("tensorcore_config")
        if block_size % self.arch.warp_size != 0:
            return None
        tile, rsteps = td.get_tile(node), td.get_rstep(node)
        warps = block_size // self.arch.warp_size
        ndim = len(tile)
        wmma = [16, 16, 16] # TODO(leiwang1999): should generalize the config
        wmma_tile = [1 for i in range(ndim)]
        wmma_tile[ax_m] = wmma[0]
        wmma_tile[ax_n] = wmma[1]
        space = [tile[i] // wmma_tile[i] for i in range(ndim)]
        if tile[ax_m] % wmma_tile[ax_m] != 0 or tile[ax_n] % wmma_tile[ax_n]:
            return None
        if np.prod(space) % warps != 0:
            return None
        factors = factorize(np.prod(space) // warps)

        def _score(node, thread): # small is better
            score = 0
            block_tile = [int(np.ceil(tile[i] / thread[i])) for i in range(ndim)]
            shape = node.propogate_inputs(block_tile)
            for i, buffer in enumerate(node.input_buffers):
                score += np.prod(shape[i]) / self.arch.bandwidth[1]
            return score

        warp_tile = wmma_tile.copy()
        for factor in reversed(factors):
            score_map = {}
            for i in range(ndim):
                if tile[i] % (warp_tile[i] * factor) != 0:
                    continue
                warp_tile[i] *= factor
                score_map[i] = (_score(node, warp_tile), i)
                warp_tile[i] //= factor
            if len(score_map) == 0:
                return None
            dim_order = sorted(score_map.keys(), key=lambda x:score_map[x])
            warp_tile[dim_order[0]] *= factor

        codegen_dict = Config()
        codegen_dict.block = tile
        codegen_dict.warp = warp_tile
        codegen_dict.use_tc = True
        codegen_dict.pipeline_stage = self.pipeline_stage
        codegen_dict.use_async = self.use_async_copy
        codegen_dict.rstep = [int(rsteps[ax.var.name]) for ax in node.raxis]
        codegen_dict.cached_tensors = td.cached_tensors_map[node]
        codegen_dict.wmma = wmma
        codegen_dict.complete_config(node)
        codegen_dict.vectorize = self._plan_vectorize(self.prim_func_node, td, block_size)
        return codegen_dict

    def plan_rasterization(self, td: TileDict):
        return NoRasterization()
