// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.

#include "flash_fwd_launch_template.h"

template<>
void run_mha_fwd_<{{cutlass_dtype}}, {{dimqk}}, {{dimv}}>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_qkdim{{dimqk}}_vdim{{dimv}}<{{cutlass_dtype}}>(params, stream);
}
