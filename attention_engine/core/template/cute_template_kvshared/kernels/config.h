#pragma once

namespace Config {

static constexpr int BLOCK_SIZE_M = 64;
static constexpr int PAGE_BLOCK_SIZE = 64;

static constexpr int HEAD_DIM_K = {{dimqk}}; // 384+64 ; // 448+64; // 576;
static constexpr int HEAD_DIM_V = {{dimv}}; // 384; // 448; // 512;

static constexpr int FIXED_OVERHEAD_NUM_BLOCKS = 5;

}
