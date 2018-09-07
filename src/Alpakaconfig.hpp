/**
 * \file
 * Copyright 2018 Sebastian Benner, Jonas Schenke
 *
 * This file is part of jungfrau-photoncounter.
 *
 * alpaka is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * alpaka is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with alpaka.
 * If not, see <http://www.gnu.org/licenses/>.
 *
 */

#pragma once

#include "iterator.hpp"
#include <alpaka/alpaka.hpp>

// Defines for dimensions and types.
using Dim = alpaka::dim::DimInt<1u>;
using Size = uint64_t;
using Extent = uint64_t;
using WorkDiv = alpaka::workdiv::WorkDivMembers<Dim, Extent>;

//#############################################################################
//! Get Trait via struct.
//!
//! \tparam T The data type.
//! \tparam TBuf The buffer type.
//! \tparam TAcc The accelerator type.
//!
//! Defines the appropriate iterator for an accelerator.
template <typename T, typename TBuf, typename TAcc>
struct GetIterator
{
    using Iterator = IteratorCpu<TAcc, T, TBuf>;
};

// Note: Boost Fibers, OpenMP 2 Threads and TBB Blocks accelerators aren't implented

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLED
struct CpuFibers {};
#endif

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED
struct CpuOmp2Threads {};
#endif

#ifdef ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
struct CpuTbbBlocks {};
#endif


#ifdef ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED
//#############################################################################
//! OpenMP 2 Blocks defines
//!
//! Defines Host, Device, etc. for the OpenMP 2 Blocks accelerator.
struct CpuOmp2Blocks
{
    using Host = alpaka::acc::AccCpuOmp2Blocks<Dim, Extent>;
    using Acc = alpaka::acc::AccCpuOmp2Blocks<Dim, Extent>;
    using DevHost = alpaka::dev::Dev<Host>;
    using DevAcc = alpaka::dev::Dev<Acc>;
    using PltfHost = alpaka::pltf::Pltf<DevHost>;
    using PltfAcc = alpaka::pltf::Pltf<DevAcc>;
    using Queue = alpaka::queue::QueueCpuSync;
    using Event = alpaka::event::Event<Queue>;
  
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

#ifdef ALPAKA_ACC_CPU_BT_OMP4_ENABLED
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
//#############################################################################
//! OpenMP 4 defines
//!
//! Defines Host, Device, etc. for the OpenMP 4 accelerator.
struct CpuOmp4
{
    using Host = alpaka::acc::AccCpuSerial<Dim, Extent>;
    using Acc = alpaka::acc::AccCpuOmp4<Dim, Extent>;
    using DevHost = alpaka::dev::Dev<Host>;
    using DevAcc = alpaka::dev::Dev<Acc>;
    using PltfHost = alpaka::pltf::Pltf<DevHost>;
    using PltfAcc = alpaka::pltf::Pltf<DevAcc>;
    using Queue = alpaka::queue::QueueCpuSync;
    using Event = alpaka::event::Event<Queue>;

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
#endif

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
//#############################################################################
//! Serial CPU defines
//!
//! Defines Host, Device, etc. for the serial CPU accelerator.
struct CpuSerial
{
    using Host = alpaka::acc::AccCpuSerial<Dim, Extent>;
    using Acc = alpaka::acc::AccCpuSerial<Dim, Extent>;
    using DevHost = alpaka::dev::Dev<Host>;
    using DevAcc = alpaka::dev::Dev<Acc>;
    using PltfHost = alpaka::pltf::Pltf<DevHost>;
    using PltfAcc = alpaka::pltf::Pltf<DevAcc>;
    using Queue = alpaka::queue::QueueCpuSync;
    using Event = alpaka::event::Event<Queue>;
  
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

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED
//#############################################################################
//! CPU Threads defines
//!
//! Defines Host, Device, etc. for the CPU Threads accelerator.
struct CpuThreads
{
    using Host = alpaka::acc::AccCpuThreads<Dim, Extent>;
    using Acc = alpaka::acc::AccCpuThreads<Dim, Extent>;
    using DevHost = alpaka::dev::Dev<Host>;
    using DevAcc = alpaka::dev::Dev<Acc>;
    using PltfHost = alpaka::pltf::Pltf<DevHost>;
    using PltfAcc = alpaka::pltf::Pltf<DevAcc>;
    using Queue = alpaka::queue::QueueCpuSync;
    using Event = alpaka::event::Event<Queue>;

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
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
//#############################################################################
//! CUDA defines
//!
//! Defines Host, Device, etc. for the CUDA accelerator.
struct GpuCudaRt
{
    using Host = alpaka::acc::AccCpuSerial<Dim, Extent>;
    using Acc = alpaka::acc::AccGpuCudaRt<Dim, Extent>;
    using DevHost = alpaka::dev::Dev<Host>;
    using DevAcc = alpaka::dev::Dev<Acc>;
    using PltfHost = alpaka::pltf::Pltf<DevHost>;
    using PltfAcc = alpaka::pltf::Pltf<DevAcc>;
    using Queue = alpaka::queue::QueueCudaRtAsync;
    using Event = alpaka::event::Event<Queue>;

    const std::size_t STREAMS_PER_DEV = 1;
  
    alpaka::vec::Vec<Dim, Size> 
        elementsPerThread =  static_cast<Size>(4u);
    alpaka::vec::Vec<Dim, Size>
        threadsPerBlock = static_cast<Size>(1024u);
    alpaka::vec::Vec<Dim, Size>
        blocksPerGrid = static_cast<Size>(512u);
    WorkDiv
        workdiv{blocksPerGrid, threadsPerBlock, elementsPerThread};};
#endif
#endif
