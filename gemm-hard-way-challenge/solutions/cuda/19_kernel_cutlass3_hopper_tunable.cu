// GEMM the Hard Way, Day 19.
// CUTLASS 3.x Hopper GEMM following:
// https://www.kapilsharma.dev/posts/learn-cutlass-the-hard-way-2/
//
// Chapter 2 implemented Hopper mechanisms directly. This lesson expresses the
// same ideas through CUTLASS 3's five-layer GEMM hierarchy:
//   Device adapter
//     -> universal kernel and tile scheduler
//       -> mainloop/epilogue collectives
//         -> tiled TMA/WGMMA operations
//           -> hardware instruction atoms
//
// CollectiveBuilder selects the TMA copies, asynchronous transaction barriers,
// producer/consumer warp roles, WGMMA operations, and epilogue implementation.
// The blanks are therefore configuration choices rather than handwritten PTX.
// Tune one group at a time and benchmark every matrix size: there is no single
// best configuration for both memory-bound small GEMMs and compute-bound large
// GEMMs.

#include <torch/torch.h>
#include <cuda_runtime.h>

#include <type_traits>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/util/packed_stride.hpp"
#include "cute/tensor.hpp"

namespace day19 {

// Schedule choices used by Blank A:
//   0: basic TMA warp specialization. One CTA owns each output tile.
//   1: cooperative persistent. Two consumer warpgroups split a tile and
//      persistent CTAs repeatedly request work.
//   2: ping-pong. The consumer warpgroups work on different tiles so one can
//      run its epilogue while the other issues WGMMA.
//   3: cooperative mainloop plus Stream-K scheduler. The decomposition knob
//      can select heuristic, data-parallel, Split-K, or Stream-K assignment.
template <int ScheduleChoice>
constexpr auto select_kernel_schedule()
{
    if constexpr (ScheduleChoice == 0) {
        return cutlass::gemm::KernelTmaWarpSpecialized{};
    } else if constexpr (ScheduleChoice == 1) {
        return cutlass::gemm::KernelTmaWarpSpecializedCooperative{};
    } else if constexpr (ScheduleChoice == 2) {
        return cutlass::gemm::KernelTmaWarpSpecializedPingpong{};
    } else {
        return cutlass::gemm::KernelTmaWarpSpecializedCooperative{};
    }
}

template <int ScheduleChoice>
constexpr auto select_epilogue_schedule()
{
    if constexpr (ScheduleChoice == 1 || ScheduleChoice == 3) {
        return cutlass::epilogue::TmaWarpSpecializedCooperative{};
    } else {
        return cutlass::epilogue::TmaWarpSpecialized{};
    }
}

template <int ScheduleChoice>
constexpr auto select_tile_scheduler()
{
    if constexpr (ScheduleChoice == 0) {
        return;
    } else if constexpr (ScheduleChoice == 3) {
        return cutlass::gemm::StreamKScheduler{};
    } else {
        return cutlass::gemm::PersistentScheduler{};
    }
}

template <int NumStages, int EpilogueCarveoutBytes>
constexpr auto select_stage_count()
{
    if constexpr (NumStages == 0) {
        return cutlass::gemm::collective::StageCountAutoCarveout<EpilogueCarveoutBytes>{};
    } else {
        return cutlass::gemm::collective::StageCount<NumStages>{};
    }
}

template <typename Scheduler, typename CollectiveMainloop, typename CollectiveEpilogue>
static auto make_gemm_kernel_type()
{
    if constexpr (std::is_void_v<Scheduler>) {
        return cutlass::gemm::kernel::GemmUniversal<
            cute::Shape<int, int, int>,
            CollectiveMainloop,
            CollectiveEpilogue>{};
    } else {
        return cutlass::gemm::kernel::GemmUniversal<
            cute::Shape<int, int, int>,
            CollectiveMainloop,
            CollectiveEpilogue,
            Scheduler>{};
    }
}

using RasterOrderOptions =
    cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90Params::RasterOrderOptions;
using DecompositionMode =
    cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90StreamKParams::DecompositionMode;

template <
    int ScheduleChoice,
    int NumStages,
    int TileChoice,
    int ClusterChoice,
    int RasterChoice,
    int DecompositionChoice,
    int MaxSwizzleSize,
    int Splits>
struct HopperGemmConfig
{
    static_assert(ScheduleChoice >= 0 && ScheduleChoice <= 3);
    static_assert(NumStages == 0 || (NumStages >= 2 && NumStages <= 5));
    static_assert(TileChoice == 0 || TileChoice == 1);
    static_assert(ClusterChoice == 0 || ClusterChoice == 1);
    static_assert(RasterChoice >= 0 && RasterChoice <= 2);
    static_assert(DecompositionChoice >= 0 && DecompositionChoice <= 3);
    static_assert(
        MaxSwizzleSize == 1 || MaxSwizzleSize == 2 ||
        MaxSwizzleSize == 4 || MaxSwizzleSize == 8);
    static_assert(Splits >= 1);

    using ElementA = cutlass::bfloat16_t;
    using ElementB = cutlass::bfloat16_t;
    using ElementC = cutlass::bfloat16_t;
    using ElementD = cutlass::bfloat16_t;
    using ElementAccumulator = float;

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::RowMajor;
    using LayoutC = cutlass::layout::RowMajor;
    using LayoutD = cutlass::layout::RowMajor;

    // TMA requires at least 16-byte aligned accesses. For BF16, 128 bits is
    // eight adjacent elements, so CUTLASS checks these alignments before launch.
    static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
    static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
    static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
    static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

    // Blank C selects 128x128x64 (0) or 128x256x64 (1). The larger N tile
    // increases arithmetic intensity and WGMMA work per CTA, but consumes more
    // registers/shared memory and can reduce occupancy on smaller problems.
    using TileShape = std::conditional_t<
        TileChoice == 0,
        cute::Shape<cute::_128, cute::_128, cute::_64>,
        cute::Shape<cute::_128, cute::_256, cute::_64>>;

    // Blank D selects no cluster (0) or the blog's 1x2x1 cluster (1).
    // The N-oriented pair can multicast the common A tile. Clustering may
    // reduce global traffic, but it also constrains placement and does not win
    // for every size; the blog's final autotune kept 1x1x1.
    using ClusterShape = std::conditional_t<
        ClusterChoice == 0,
        cute::Shape<cute::_1, cute::_1, cute::_1>,
        cute::Shape<cute::_1, cute::_2, cute::_1>>;

    using KernelSchedule = decltype(select_kernel_schedule<ScheduleChoice>());
    using EpilogueSchedule = decltype(select_epilogue_schedule<ScheduleChoice>());
    using TileSchedulerType = decltype(select_tile_scheduler<ScheduleChoice>());

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90,
        cutlass::arch::OpClassTensorOp,
        TileShape,
        ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator,
        ElementAccumulator,
        ElementC, LayoutC, AlignmentC,
        ElementD, LayoutD, AlignmentD,
        EpilogueSchedule>::CollectiveOp;

    // Blank B uses 0 for automatic stage selection or 2-5 for a fixed pipeline.
    // More stages hide TMA latency but reserve more shared memory. The epilogue
    // carveout is subtracted before CUTLASS computes an automatic stage count.
    using StageCount = decltype(select_stage_count<
        NumStages,
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>());

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90,
        cutlass::arch::OpClassTensorOp,
        ElementA, LayoutA, AlignmentA,
        ElementB, LayoutB, AlignmentB,
        ElementAccumulator,
        TileShape,
        ClusterShape,
        StageCount,
        KernelSchedule>::CollectiveOp;

    using GemmKernel = decltype(
        make_gemm_kernel_type<TileSchedulerType, CollectiveMainloop, CollectiveEpilogue>());
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    // Blanks E-H are consumed by StreamKScheduler. Raster values are
    // 0=AlongM, 1=AlongN, 2=Heuristic. Decomposition values are
    // 0=Heuristic, 1=DataParallel, 2=SplitK, 3=StreamK.
    static constexpr RasterOrderOptions RasterOrder =
        static_cast<RasterOrderOptions>(RasterChoice);
    static constexpr DecompositionMode Decomposition =
        static_cast<DecompositionMode>(DecompositionChoice);
    static constexpr int Swizzle = MaxSwizzleSize;
    static constexpr int SplitCount = Splits;
};

template <typename Scheduler>
struct is_streamk_scheduler : std::false_type {};

template <>
struct is_streamk_scheduler<cutlass::gemm::StreamKScheduler> : std::true_type {};

template <typename Config>
cudaError_t launch(
    int M,
    int N,
    int K,
    const typename Config::ElementA *d_A,
    const typename Config::ElementB *d_B,
    typename Config::ElementD *d_D,
    cudaStream_t stream = nullptr)
{
    if (M == 0 || N == 0 || K == 0)
        return cudaSuccess;

    typename Config::Gemm gemm_op;
    auto problem_shape = cute::make_shape(M, N, K);

    using StrideA = typename Config::GemmKernel::StrideA;
    using StrideB = typename Config::GemmKernel::StrideB;
    using StrideC = typename Config::GemmKernel::StrideC;
    using StrideD = typename Config::GemmKernel::StrideD;

    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
    auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count =
        cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

    typename Config::Gemm::Arguments args = [&]() {
        if constexpr (is_streamk_scheduler<typename Config::TileSchedulerType>::value) {
            typename Config::GemmKernel::TileScheduler::Arguments scheduler_args{
                Config::SplitCount,
                Config::Swizzle,
                Config::RasterOrder,
                Config::Decomposition
            };
            return typename Config::Gemm::Arguments{
                cutlass::gemm::GemmUniversalMode::kGemm,
                problem_shape,
                {d_A, stride_A, d_B, stride_B},
                {{1.0f, 0.0f}, d_D, stride_C, d_D, stride_D},
                hw_info,
                scheduler_args
            };
        } else {
            return typename Config::Gemm::Arguments{
                cutlass::gemm::GemmUniversalMode::kGemm,
                problem_shape,
                {d_A, stride_A, d_B, stride_B},
                {{1.0f, 0.0f}, d_D, stride_C, d_D, stride_D},
                hw_info
            };
        }
    }();

    cutlass::Status status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess)
        return cudaErrorNotSupported;

    const size_t workspace_size = Config::Gemm::get_workspace_size(args);
    void *workspace = nullptr;
    if (workspace_size > 0) {
        cudaError_t result = cudaMalloc(&workspace, workspace_size);
        if (result != cudaSuccess)
            return result;
    }

    status = gemm_op.initialize(args, workspace, stream);
    if (status == cutlass::Status::kSuccess)
        status = gemm_op.run(stream);

    if (workspace)
        cudaFree(workspace);

    return status == cutlass::Status::kSuccess ? cudaSuccess : cudaErrorUnknown;
}

// Blanks A-H - the configuration to benchmark:
//
// A, schedule:      0 basic, 1 persistent, 2 ping-pong, 3 Stream-K capable.
// B, stages:        0 auto, or a fixed value from 2 through 5.
// C, tile:          0 = 128x128x64, 1 = 128x256x64.
// D, cluster:       0 = 1x1x1, 1 = 1x2x1.
// E, raster:        0 AlongM, 1 AlongN, 2 Heuristic.
// F, decomposition: 0 Heuristic, 1 DataParallel, 2 SplitK, 3 StreamK.
// G, swizzle:       maximum swizzle size: 1, 2, 4, or 8.
// H, splits:        at least 1; relevant to Split-K/Stream-K decomposition.
//
// Begin with the basic/auto/small-tile/no-cluster configuration to establish a
// Hopper baseline. Then reproduce the blog's order: fixed versus auto stages,
// cluster shape, persistent cooperative, ping-pong, Stream-K, and finally the
// raster/decomposition/swizzle search. Change one dimension per benchmark.
using Day19Config = HopperGemmConfig<
    3,
    3,
    1,
    0,
    2,
    0,
    1,
    1>;

} // namespace day19

void sgemm_cutlass3_hopper_tunable_bf16(
    const torch::Tensor &matrix_a,
    const torch::Tensor &matrix_b,
    torch::Tensor &output_matrix)
{
    TORCH_CHECK(matrix_a.device().is_cuda(), "Matrix A must be on CUDA device");
    TORCH_CHECK(matrix_b.device().is_cuda(), "Matrix B must be on CUDA device");
    TORCH_CHECK(output_matrix.device().is_cuda(), "Output matrix must be on CUDA device");
    TORCH_CHECK(matrix_a.scalar_type() == at::kBFloat16, "Matrix A must be bfloat16");
    TORCH_CHECK(matrix_b.scalar_type() == at::kBFloat16, "Matrix B must be bfloat16");
    TORCH_CHECK(output_matrix.scalar_type() == at::kBFloat16, "Output matrix must be bfloat16");
    TORCH_CHECK(matrix_a.dim() == 2 && matrix_b.dim() == 2, "A and B must be 2D tensors");
    TORCH_CHECK(matrix_a.is_contiguous() && matrix_b.is_contiguous(),
                "Input tensors must be contiguous for TMA");
    TORCH_CHECK(output_matrix.is_contiguous(), "Output tensor must be contiguous");

    const int M = static_cast<int>(matrix_a.size(0));
    const int K = static_cast<int>(matrix_a.size(1));
    const int N = static_cast<int>(matrix_b.size(1));
    TORCH_CHECK(matrix_b.size(0) == K, "Matrix dimension mismatch");
    TORCH_CHECK(output_matrix.size(0) == M && output_matrix.size(1) == N,
                "Output matrix has wrong shape");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(matrix_a.data_ptr()) % 16 == 0,
                "Matrix A must be 16-byte aligned for Hopper TMA");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(matrix_b.data_ptr()) % 16 == 0,
                "Matrix B must be 16-byte aligned for Hopper TMA");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(output_matrix.data_ptr()) % 16 == 0,
                "Output matrix must be 16-byte aligned for Hopper TMA");

    // Blank I - completion gate:
    // Keep this false until A-H are explicit choices. The placeholder values
    // form a compilable baseline configuration, but returning zeros prevents
    // accidentally treating unfilled tuning knobs as measured results.
    if (!1) {
        output_matrix.zero_();
        return;
    }

    const auto *d_A =
        reinterpret_cast<const cutlass::bfloat16_t *>(matrix_a.data_ptr<at::BFloat16>());
    const auto *d_B =
        reinterpret_cast<const cutlass::bfloat16_t *>(matrix_b.data_ptr<at::BFloat16>());
    auto *d_D =
        reinterpret_cast<cutlass::bfloat16_t *>(output_matrix.data_ptr<at::BFloat16>());

    const cudaError_t error = day19::launch<day19::Day19Config>(M, N, K, d_A, d_B, d_D);
    TORCH_CHECK(
        error == cudaSuccess,
        "Day 19 CUTLASS 3 Hopper GEMM failed: ",
        cudaGetErrorString(error));
}
