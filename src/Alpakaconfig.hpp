#pragma once

#include "Config.hpp"


#ifdef ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLED
struct CpuFibers {};
#endif

#ifdef ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED
struct CpuOmp2Blocks {};
#endif

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED
struct CpuOmp2Threads {};
#endif

#ifdef ALPAKA_ACC_CPU_BT_OMP4_ENABLED
struct CpuOmp4 {};
#endif

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
struct CpuSerial {
    using Dim = alpaka::dim::DimInt<1u>;
    using Size = std::size_t;
    using Host = alpaka::acc::AccCpuSerial<Dim, Size>;
    using Acc = alpaka::acc::AccCpuSerial<Dim, Size>;
    using DevHost = alpaka::dev::Dev<Host>;
    using DevAcc = alpaka::dev::Dev<Acc>;
    using PltfHost = alpaka::pltf::Pltf<DevHost>;
    using PltfAcc = alpaka::pltf::Pltf<DevAcc>;
    using WorkDiv = alpaka::workdiv::WorkDivMembers<Dim, Size>;
    using Stream = alpaka::stream::StreamCpuSync;

    alpaka::vec::Vec<Dim, Size> const elementsPerThread(static_cast<Size>(1u));
    alpaka::vec::Vec<Dim, Size> const threadsPerBlock(static_cast<Size>(1u));
    alpaka::vec::Vec<Dim, Size> const blocksPerGrid(static_cast<Size>(MAPSIZE));
    WorkDiv const workdiv(blocksPerGrid, threadsPerBlock, elementsPerThread);
};
#endif

#ifdef ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
struct CpuTbbBlocks {};
#endif

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED
struct CpuThreads {};
#endif

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
struct GpuCudaRt {};
#endif