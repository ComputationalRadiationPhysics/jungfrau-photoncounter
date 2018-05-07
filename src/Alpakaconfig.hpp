#pragma once

#include "Config.hpp"


#ifdef ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLED
struct CpuFibers {};
#endif

#ifdef ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED
struct CpuOmp2Blocks {
    using Dim = alpaka::dim::DimInt<1u>;
    using Size = std::size_t;
    using Host = alpaka::acc::AccCpuOmp2Blocks<Dim, Size>;
    using Acc = alpaka::acc::AccCpuOmp2Blocks<Dim, Size>;
    using DevHost = alpaka::dev::Dev<Host>;
    using DevAcc = alpaka::dev::Dev<Acc>;
    using PltfHost = alpaka::pltf::Pltf<DevHost>;
    using PltfAcc = alpaka::pltf::Pltf<DevAcc>;
    using WorkDiv = alpaka::workdiv::WorkDivMembers<Dim, Size>;
    using Stream = alpaka::stream::StreamCpuSync;
    using Event = alpaka::event::Event<Stream>;

    const std::size_t STREAMS_PER_DEV = 4;

    alpaka::vec::Vec<Dim, Size> 
        elementsPerThread =  static_cast<Size>(1u);
    alpaka::vec::Vec<Dim, Size>
        threadsPerBlock = static_cast<Size>(1u);
    alpaka::vec::Vec<Dim, Size>
        blocksPerGrid = static_cast<Size>(MAPSIZE);
    WorkDiv
        workdiv{blocksPerGrid, threadsPerBlock, elementsPerThread};
};
#endif

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED
struct CpuOmp2Threads {};
#endif

#ifdef ALPAKA_ACC_CPU_BT_OMP4_ENABLED
struct CpuOmp4 {
    using Dim = alpaka::dim::DimInt<1u>;
    using Size = std::size_t;
    using Host = alpaka::acc::AccCpuSerial<Dim, Size>;
    using Acc = alpaka::acc::AccCpuOmp4<Dim, Size>;
    using DevHost = alpaka::dev::Dev<Host>;
    using DevAcc = alpaka::dev::Dev<Acc>;
    using PltfHost = alpaka::pltf::Pltf<DevHost>;
    using PltfAcc = alpaka::pltf::Pltf<DevAcc>;
    using WorkDiv = alpaka::workdiv::WorkDivMembers<Dim, Size>;
    using Stream = alpaka::stream::StreamCpuSync;
    using Event = alpaka::event::Event<Stream>;

    const std::size_t STREAMS_PER_DEV = 4;

    alpaka::vec::Vec<Dim, Size> 
        elementsPerThread =  static_cast<Size>(1u);
    alpaka::vec::Vec<Dim, Size>
        threadsPerBlock = static_cast<Size>(1u);
    alpaka::vec::Vec<Dim, Size>
        blocksPerGrid = static_cast<Size>(MAPSIZE);
    WorkDiv
        workdiv{blocksPerGrid, threadsPerBlock, elementsPerThread};
};
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
    using Event = alpaka::event::Event<Stream>;

    const std::size_t STREAMS_PER_DEV = 4;

    alpaka::vec::Vec<Dim, Size> 
        elementsPerThread =  static_cast<Size>(1u);
    alpaka::vec::Vec<Dim, Size>
        threadsPerBlock = static_cast<Size>(1u);
    alpaka::vec::Vec<Dim, Size>
        blocksPerGrid = static_cast<Size>(MAPSIZE);
    WorkDiv
        workdiv{blocksPerGrid, threadsPerBlock, elementsPerThread};
};
#endif

#ifdef ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
struct CpuTbbBlocks {};
#endif

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED
struct CpuThreads {
    using Dim = alpaka::dim::DimInt<1u>;
    using Size = std::size_t;
    using Host = alpaka::acc::AccCpuThreads<Dim, Size>;
    using Acc = alpaka::acc::AccCpuThreads<Dim, Size>;
    using DevHost = alpaka::dev::Dev<Host>;
    using DevAcc = alpaka::dev::Dev<Acc>;
    using PltfHost = alpaka::pltf::Pltf<DevHost>;
    using PltfAcc = alpaka::pltf::Pltf<DevAcc>;
    using WorkDiv = alpaka::workdiv::WorkDivMembers<Dim, Size>;
    using Stream = alpaka::stream::StreamCpuSync;
    using Event = alpaka::event::Event<Stream>;

    const std::size_t STREAMS_PER_DEV = 4;

    alpaka::vec::Vec<Dim, Size> 
        elementsPerThread =  static_cast<Size>(1u);
    alpaka::vec::Vec<Dim, Size>
        threadsPerBlock = static_cast<Size>(1u);
    alpaka::vec::Vec<Dim, Size>
        blocksPerGrid = static_cast<Size>(MAPSIZE);
    WorkDiv
        workdiv{blocksPerGrid, threadsPerBlock, elementsPerThread};
};
#endif

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
struct GpuCudaRt {
    using Dim = alpaka::dim::DimInt<1u>;
    using Size = std::size_t;
    using Host = alpaka::acc::AccCpuSerial<Dim, Size>;
    using Acc = alpaka::acc::AccGpuCudaRt<Dim, Size>;
    using DevHost = alpaka::dev::Dev<Host>;
    using DevAcc = alpaka::dev::Dev<Acc>;
    using PltfHost = alpaka::pltf::Pltf<DevHost>;
    using PltfAcc = alpaka::pltf::Pltf<DevAcc>;
    using WorkDiv = alpaka::workdiv::WorkDivMembers<Dim, Size>;
    using Stream = alpaka::stream::StreamCudaRtAsync;
    using Event = alpaka::event::Event<Stream>;
    
    const std::size_t STREAMS_PER_DEV = 1;

    alpaka::vec::Vec<Dim, Size> 
        elementsPerThread =  static_cast<Size>(4u);
    alpaka::vec::Vec<Dim, Size>
        threadsPerBlock = static_cast<Size>(1024u);
    alpaka::vec::Vec<Dim, Size>
        blocksPerGrid = static_cast<Size>(512u);
    WorkDiv
        workdiv{blocksPerGrid, threadsPerBlock, elementsPerThread};
};
#endif
