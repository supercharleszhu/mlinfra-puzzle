// GEMM the Hard Way solution edition.
// Day 13 focus: handwritten fast.cu matmul_2-style Hopper TMA + WGMMA.
//
// Read the reference:
//   https://github.com/pranjalssh/fast.cu/blob/main/examples/matmul/matmul_2.cuh
//
// This day follows the Day 12 Tensor Core async baseline and starts the
// handwritten Hopper chapter. The goal is to manually identify the Hopper
// pieces that library examples often hide:
//   1. encode WGMMA shared-memory descriptors,
//   2. create 2D TMA tensor maps for A and B^T,
//   3. issue cp.async.bulk.tensor global-to-shared copies,
//   4. run wgmma.m64n64k16 over four BK=16 slices,
//   5. store the 64x64 accumulator tile into C^T.
//
// Transition from Day 12 to Day 13:
//   Problem names:
//     num_rows_a -> M, num_cols_b -> N, num_cols_a -> K
//     matrix_a -> A, matrix_b -> B^T, matrix_c/output_matrix -> C^T
//   Shared-memory names:
//     tile_a/tile_b -> sA/sB
//   Copy path:
//     Day 12: each thread computes global addresses and calls cuda::memcpy_async.
//     Day 13: one thread issues cp.async.bulk.tensor_2d using a TMA tensor map,
//             then all 128 threads wait on a cuda::barrier.
//   Tensor Core path:
//     Day 12: nvcuda::wmma fragments + nvcuda::wmma::mma_sync, one warp.
//     Day 13: inline PTX wgmma.mma_async + shared-memory descriptors, one
//             128-thread warpgroup.
//   Tile naming:
//     WMMA_M/WMMA_N/WMMA_K -> WGMMA_M/WGMMA_N/WGMMA_K
//     Day 12 used BM=256, BN=128, BK=16.
//     Day 13 uses BM=64, BN=64, BK=64 and covers BK with four WGMMA_K=16 ops.
//   Output:
//     Day 12 stores row-major FP32 C.
//     Day 13 stores BF16 C^T because the fast.cu handwritten path uses B^T/C^T.

using std::max;
typedef __nv_bfloat16 bf16;

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

// Builds the WGMMA shared-memory descriptor operand for A/B tiles.
//
// day13_make_smem_desc() turns a shared-memory pointer into the descriptor
// operand passed to wgmma.mma_async. WGMMA does not take normal pointers for A/B;
// it takes a packed descriptor containing the smem base address, layout strides,
// and swizzle mode. For matmul_2's 64x64 BF16 tile, the filled values are:
//   day13_matrix_descriptor_encode(x) = ((x & 0x3FFFF) >> 4)
//   leading byte offset              = 16
//   stride byte offset               = 1024
//   swizzle bit                      = 1
//
// Descriptor layout, from the PTX WGMMA matrix-descriptor format:
//   bits  0..15: shared-memory base address, encoded in 16-byte units
//   bits 16..31: leading dimension byte offset, encoded in 16-byte units
//   bits 32..47: stride byte offset, encoded in 16-byte units
//   bit      62: 128-byte swizzle selector
//
// `__cvta_generic_to_shared(ptr)` converts the generic pointer to a shared-memory
// address. WGMMA descriptors only keep the low shared-memory address bits, so
// `x & 0x3FFFF` keeps the descriptor-visible address range and `>> 4` converts
// bytes to 16-byte units. In this 64x64 BF16 tile:
//   16 bytes  = distance between adjacent K-slices consumed by the GMMA atom
//   1024 bytes = 64 * 8 * sizeof(bf16), the stride used by the swizzled SMEM layout
//
// Concrete example:
//   if __cvta_generic_to_shared(ptr) returns 0x1000, then
//     base    = day13_matrix_descriptor_encode(0x1000) = 0x100
//     leading = day13_matrix_descriptor_encode(16)     = 0x1
//     stride  = day13_matrix_descriptor_encode(1024)   = 0x40
//   The packed descriptor is:
//     desc = base | (leading << 16) | (stride << 32) | (1ULL << 62)
//          = 0x100 | 0x0000000000010000 | 0x0000004000000000
//            | 0x4000000000000000
//          = 0x4000004000010100
// The exact base changes with the shared-memory address, but the leading,
// stride, and swizzle fields are fixed for this Day 13 layout.
__device__ static inline uint64_t day13_matrix_descriptor_encode(uint64_t x) {
    return ((x & 0x3FFFF) >> 0x4);
}

__device__ uint64_t day13_make_smem_desc(bf16* ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    uint64_t desc = day13_matrix_descriptor_encode(addr);
    desc |= day13_matrix_descriptor_encode(static_cast<uint64_t>(16)) << 16;
    desc |= day13_matrix_descriptor_encode(static_cast<uint64_t>(1024)) << 32;
    desc |= 1llu << 62; // 128B swizzle.
    return desc;
}

__device__ void day13_warpgroup_arrive() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ void day13_warpgroup_commit_batch() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

template <int N>
__device__ void day13_warpgroup_wait() {
    static_assert(N >= 0 && N <= 7, "WGMMA wait: N must be in range [0, 7]");
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}

// Creates the 2D tensor map consumed by cp.async.bulk.tensor.
//
// A CUtensorMap is the address/shape/stride recipe consumed by Hopper's TMA
// engine. It describes the full tensor in global memory and the rectangular
// "box" to copy into shared memory. After this map is encoded, the kernel can
// issue one cp.async.bulk.tensor_2d instruction with tile coordinates instead
// of having many threads compute individual global-memory addresses.
//
// Flow:
//   CUtensorMap      -> describes A or B^T in global memory
//   TMA copy         -> uses the map to load one tile into sA or sB
//   WGMMA descriptor -> describes the loaded shared-memory tile to tensor cores
//   WGMMA            -> computes from sA/sB
//
// The tensor map describes global memory so TMA can copy rectangular tiles into
// shared memory without per-thread address arithmetic. In matmul_2, for template
// arguments <BlockMajorSize, BlockMinorSize>:
//   gmem_prob_shape = {BlockMinorSize * blocks_width,
//                      BlockMajorSize * blocks_height, 1, 1, 1}
//   gmem_prob_stride = {sizeof(bf16),
//                       sizeof(bf16) * BlockMinorSize * blocks_width, 0, 0, 0}
//   smem_box_shape = {BlockMinorSize, BlockMajorSize, 1, 1, 1}
// For A, this becomes <BM, BK>; for B^T, this becomes <BN, BK>.
//
// Naming:
//   BlockMajorSize = number of rows in one tile along the matrix's major axis.
//                    For A this is BM; for B^T this is BN.
//   BlockMinorSize = number of columns in one tile along the matrix's minor axis.
//                    For both A and B^T this is BK.
//   blocks_height  = number of BlockMajorSize tiles covering the full matrix
//                    height, e.g. M / BM for A or N / BN for B^T.
//   blocks_width   = number of BlockMinorSize tiles covering the full matrix
//                    width, e.g. K / BK.
//
// Example for A with M=4096, K=4096, BM=64, BK=64:
//   BlockMajorSize=64, BlockMinorSize=64, blocks_height=64, blocks_width=64
//   gmem_prob_shape={4096, 4096, 1, 1, 1}
// Example for B^T with N=4096, K=4096, BN=64, BK=64:
//   BlockMajorSize=64, BlockMinorSize=64, blocks_height=64, blocks_width=64
//   gmem_prob_shape={4096, 4096, 1, 1, 1}
template <int BlockMajorSize, int BlockMinorSize>
void day13_create_tensor_map(CUtensorMap *tma_map, bf16* gmem_ptr, int blocks_height, int blocks_width) {
    void* gmem_address = static_cast<void*>(gmem_ptr);
    uint64_t gmem_prob_shape[5] = {
        static_cast<uint64_t>(BlockMinorSize * blocks_width),
        static_cast<uint64_t>(BlockMajorSize * blocks_height),
        1, 1, 1};
    uint64_t gmem_prob_stride[5] = {
        sizeof(bf16),
        sizeof(bf16) * static_cast<uint64_t>(BlockMinorSize * blocks_width),
        0, 0, 0};
    uint32_t smem_box_shape[5] = {
        uint32_t(BlockMinorSize),
        uint32_t(BlockMajorSize),
        1, 1, 1};
    uint32_t smem_box_stride[5] = {1, 1, 1, 1, 1};

    CUresult result = cuTensorMapEncodeTiled(
        tma_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, gmem_address, gmem_prob_shape,
        gmem_prob_stride + 1, smem_box_shape, smem_box_stride, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    assert(result == CUDA_SUCCESS);
}

CUtensorMap *d_tma_map_A = nullptr;
CUtensorMap *d_tma_map_B = nullptr;
int _prev_m = 0, _prev_n = 0, _prev_k = 0;

template <int BlockMajorSize, int BlockMinorSize>
__host__ static inline CUtensorMap* day13_allocate_and_create_tensor_map(
    bf16* src,
    int blocks_height,
    int blocks_width)
{
    CUtensorMap *tma_map_d;
    cudaMalloc(&tma_map_d, sizeof(CUtensorMap));
    CUtensorMap tma_map_host;
    day13_create_tensor_map<BlockMajorSize, BlockMinorSize>(
        &tma_map_host, src, blocks_height, blocks_width);
    cudaMemcpy(tma_map_d, &tma_map_host, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
    return tma_map_d;
}

// Hopper WGMMA has no `nvcuda::wgmma` C++ API equivalent to `nvcuda::wmma`.
// The handwritten path emits the exact WGMMA PTX instruction and passes shared
// memory descriptors for A/B; CUTLASS/CuTe provide higher-level wrappers that
// generate similar WGMMA instructions internally.
//
// `d` is this thread's FP32 accumulator fragment for the 64x64 output tile.
// The whole warpgroup owns 64 * 64 = 4096 FP32 accumulators; with 128 threads,
// that is 32 registers per thread. The shape d[4][8] is just a convenient way
// to name those 32 registers: four 16-column groups, eight per group. WGMMA
// updates d in place, and the later store code maps each d[w][0..7] value to
// its C^T element.
//
// The template parameters are PTX immediates:
//   ScaleD=1 keeps the old accumulator value and computes D = A*B + D.
//   ScaleA=1 and ScaleB=1 use A and B normally.
//   TransA=0 and TransB=0 mean the shared-memory descriptors already present A
//   and B in the orientation expected by the instruction.
// Example:
//   day13_wgmma64<1, 1, 1, 0, 0>(d, &sA[0], &sB[0]);
// accumulates one m64n64k16 slice from sA/sB into d.
template<int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
__device__ void day13_wgmma64(float d[4][8], bf16* sA, bf16* sB) {
    uint64_t desc_a = day13_make_smem_desc(&sA[0]);
    uint64_t desc_b = day13_make_smem_desc(&sB[0]);
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31},"
        " %32,"
        " %33,"
        " %34, %35, %36, %37, %38;\n"
        "}\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]),
          "+f"(d[0][4]), "+f"(d[0][5]), "+f"(d[0][6]), "+f"(d[0][7]),
          "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
          "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]),
          "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
          "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
          "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]),
          "+f"(d[3][4]), "+f"(d[3][5]), "+f"(d[3][6]), "+f"(d[3][7])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
          "n"(int32_t(ScaleB)), "n"(int32_t(TransA)), "n"(int32_t(TransB)));
}

template<int BM, int BN, int BK, int WGMMA_M, int WGMMA_N, int WGMMA_K, int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS) day13_matmul_kernel(
    int M,
    int N,
    int K,
    bf16* C,
    const CUtensorMap* tensorMapA,
    const CUtensorMap* tensorMapB)
{
    __shared__ alignas(128) bf16 sA[BM * BK];
    __shared__ alignas(128) bf16 sB[BK * BN];
    // Per-thread FP32 accumulator registers for the WGMMA output tile.
    // 64*64 output values are distributed across 128 warpgroup threads, giving
    // each thread 32 accumulator registers. WGMMA groups them by 16 columns of
    // N: WGMMA_N/16 groups, 8 values per group. With WGMMA_N=64, d is d[4][8].
    float d[WGMMA_N / 16][8];
    static_assert(sizeof(d) * 128 == BM * BN * sizeof(float));
    memset(d, 0, sizeof(d));

    const int num_k_tiles = K / BK;
    // blockIdx.x is flattened over output tiles. These names are tile
    // coordinates, not counts: block_iter_m selects the M/BM tile row and
    // block_iter_n selects the N/BN tile column. block_iter_k below selects the
    // current K/BK slice loaded by TMA.
    int block_iter_n = blockIdx.x % (N / BN);
    int block_iter_m = blockIdx.x / (N / BN);
    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier barA;
    __shared__ barrier barB;

    if (threadIdx.x == 0) {
        init(&barA, blockDim.x);
        init(&barB, blockDim.x);
        cde::fence_proxy_async_shared_cta();
    }
    __syncthreads();

    barrier::arrival_token tokenA, tokenB;
    for (int block_iter_k = 0; block_iter_k < num_k_tiles; ++block_iter_k) {
        if (threadIdx.x == 0) {
            // Thread 0 issues the TMA copies. tensorMapA/tensorMapB describe the
            // full tensors; the coordinates below pick the current tile:
            //   A:   (block_iter_k * BK, block_iter_m * BM)
            //   B^T: (block_iter_k * BK, block_iter_n * BN)
            // The first coordinate walks K, the second selects the output tile.
            cde::cp_async_bulk_tensor_2d_global_to_shared(
                &sA[0], tensorMapA, block_iter_k * BK, block_iter_m * BM, barA);
            tokenA = cuda::device::barrier_arrive_tx(barA, 1, sizeof(sA));
            cde::cp_async_bulk_tensor_2d_global_to_shared(
                &sB[0], tensorMapB, block_iter_k * BK, block_iter_n * BN, barB);
            tokenB = cuda::device::barrier_arrive_tx(barB, 1, sizeof(sB));
        } else {
            // Other threads do not issue TMA, but they still arrive at the
            // same barriers. The barrier waits for all threads and for the TMA
            // transaction byte count before WGMMA consumes sA/sB.
            tokenA = barA.arrive();
            tokenB = barB.arrive();
        }
        barA.wait(std::move(tokenA));
        barB.wait(std::move(tokenB));
        __syncthreads();

        day13_warpgroup_arrive();
        // One 128-thread warpgroup owns this 64x64 output tile. The WGMMA
        // instruction shape is m64n64k16, so each call consumes one K-slice of
        // width WGMMA_K=16 from the already-loaded BK=64 shared-memory tile.
        // Therefore BK/WGMMA_K = 4 calls cover the whole reduction chunk. If
        // BK changed, this would need a loop or a different number of calls;
        // for BF16 Hopper WGMMA, the instruction K dimension is fixed at 16.
        day13_wgmma64<1, 1, 1, 0, 0>(d, &sA[0], &sB[0]);
        day13_wgmma64<1, 1, 1, 0, 0>(d, &sA[WGMMA_K], &sB[WGMMA_K]);
        day13_wgmma64<1, 1, 1, 0, 0>(d, &sA[2 * WGMMA_K], &sB[2 * WGMMA_K]);
        day13_wgmma64<1, 1, 1, 0, 0>(d, &sA[3 * WGMMA_K], &sB[3 * WGMMA_K]);
        day13_warpgroup_commit_batch();
        day13_warpgroup_wait<0>();
    }

    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp = tid / 32;
    uint32_t row = warp * 16 + lane / 4;
    bf16 *block_C = C + block_iter_n * BN * M + block_iter_m * BM;

    // matmul_2 uses direct per-thread stores here. Each thread maps its eight
    // FP32 accumulator values through __float2bfloat16 into C^T positions like:
    //   block_C[col * M + row]       = d[w][0]
    //   block_C[(col + 1) * M + row] = d[w][1]
    // with row+8 and col+8 variants for the rest of the values.
    for (int m_it = 0; m_it < BM / WGMMA_M; ++m_it) {
        for (int n_it = 0; n_it < BN / WGMMA_N; ++n_it) {
            for (int w = 0; w < WGMMA_N / 16; ++w) {
                int col = 16 * w + 2 * (tid % 4);
                #define IDX(i, j) ((j + n_it * WGMMA_N) * M + ((i) + m_it * WGMMA_M))
                block_C[IDX(row, col)] = __float2bfloat16(d[w][0]);
                block_C[IDX(row, col + 1)] = __float2bfloat16(d[w][1]);
                block_C[IDX(row + 8, col)] = __float2bfloat16(d[w][2]);
                block_C[IDX(row + 8, col + 1)] = __float2bfloat16(d[w][3]);
                block_C[IDX(row, col + 8)] = __float2bfloat16(d[w][4]);
                block_C[IDX(row, col + 9)] = __float2bfloat16(d[w][5]);
                block_C[IDX(row + 8, col + 8)] = __float2bfloat16(d[w][6]);
                block_C[IDX(row + 8, col + 9)] = __float2bfloat16(d[w][7]);
                #undef IDX
            }
        }
    }
}

void day13_run_kernel(int M, int N, int K, bf16 *A, bf16 *B, bf16 *C) {
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 64;
    constexpr int NUM_THREADS = 128;

    if (!d_tma_map_A || M != _prev_m || N != _prev_n || K != _prev_k) {
        d_tma_map_A = day13_allocate_and_create_tensor_map<BM, BK>(A, M / BM, K / BK);
        d_tma_map_B = day13_allocate_and_create_tensor_map<BN, BK>(B, N / BN, K / BK);
        _prev_m = M;
        _prev_n = N;
        _prev_k = K;
    }

    day13_matmul_kernel<
        /*BM*/ BM,
        /*BN*/ BN,
        /*BK*/ BK,
        /*WGMMA_M*/ 64,
        /*WGMMA_N*/ 64,
        /*WGMMA_K*/ 16,
        /*NUM_THREADS*/ NUM_THREADS>
        <<<(M / BM) * (N / BN), NUM_THREADS>>>(M, N, K, C, d_tma_map_A, d_tma_map_B);
}

void sgemm_fastcu_matmul2_manual_tma_wgmma_bf16(
    const torch::Tensor &matrix_a,
    const torch::Tensor &matrix_b_transposed,
    torch::Tensor &output_matrix_transposed)
{
    TORCH_CHECK(matrix_a.device().is_cuda(), "Matrix A must be on CUDA device");
    TORCH_CHECK(matrix_b_transposed.device().is_cuda(), "Matrix B^T must be on CUDA device");
    TORCH_CHECK(output_matrix_transposed.device().is_cuda(), "Output C^T must be on CUDA device");

    TORCH_CHECK(matrix_a.scalar_type() == at::kBFloat16, "Matrix A must be bfloat16");
    TORCH_CHECK(matrix_b_transposed.scalar_type() == at::kBFloat16, "Matrix B^T must be bfloat16");
    TORCH_CHECK(output_matrix_transposed.scalar_type() == at::kBFloat16, "Output C^T must be bfloat16");

    TORCH_CHECK(matrix_a.dim() == 2 && matrix_b_transposed.dim() == 2, "A and B^T must be 2D tensors");
    TORCH_CHECK(matrix_a.is_contiguous() && matrix_b_transposed.is_contiguous(),
                "A and B^T must be contiguous");
    TORCH_CHECK(output_matrix_transposed.is_contiguous(), "Output C^T must be contiguous");

    const int M = static_cast<int>(matrix_a.size(0));
    const int K = static_cast<int>(matrix_a.size(1));
    const int N = static_cast<int>(matrix_b_transposed.size(0));
    TORCH_CHECK(M % 64 == 0 && N % 64 == 0 && K % 64 == 0,
                "Day 13 matmul_2 requires M, N, and K to be multiples of 64");
    TORCH_CHECK(matrix_b_transposed.size(1) == K, "B^T must have shape N x K");
    TORCH_CHECK(output_matrix_transposed.size(0) == N && output_matrix_transposed.size(1) == M,
                "Output C^T must have shape N x M");

    auto *d_A = reinterpret_cast<bf16 *>(matrix_a.data_ptr<at::BFloat16>());
    auto *d_Bt = reinterpret_cast<bf16 *>(matrix_b_transposed.data_ptr<at::BFloat16>());
    auto *d_Ct = reinterpret_cast<bf16 *>(output_matrix_transposed.data_ptr<at::BFloat16>());

    day13_run_kernel(M, N, K, d_A, d_Bt, d_Ct);
}
