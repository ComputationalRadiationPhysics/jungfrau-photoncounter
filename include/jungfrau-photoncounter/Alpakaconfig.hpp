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

#include "Config.hpp"
#include "kernel/Iterator.hpp"
#include <alpaka/alpaka.hpp>

// Defines workdiv.
using WorkDiv = alpaka::workdiv::WorkDivMembers<Dim, Size>;
using CpuSyncQueue = alpaka::queue::QueueCpuSync;

template <typename TAccelerator> WorkDiv getWorkDiv()
{
    return WorkDiv(
        decltype(TAccelerator::blocksPerGrid)(TAccelerator::blocksPerGrid),
        decltype(TAccelerator::threadsPerBlock)(TAccelerator::threadsPerBlock),
        decltype(TAccelerator::elementsPerThread)(
            TAccelerator::elementsPerThread));
}

//#############################################################################
//! Get Trait via struct.
//!
//! \tparam T The data type.
//! \tparam TBuf The buffer type.
//! \tparam TAcc The accelerator type.
//!
//! Defines the appropriate iterator for an accelerator.
template <typename T, typename TBuf, typename TAcc> struct GetIterator {
    using Iterator = IteratorCpu<TAcc, T, TBuf>;
};

// Note: Boost Fibers, OpenMP 2 Threads and TBB Blocks accelerators aren't
// implented

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLED
struct CpuFibers {
};
#endif

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED
struct CpuOmp2Threads {
};
#endif

#ifdef ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
struct CpuTbbBlocks {
};
#endif

template <typename THost, typename TAcc, typename TQueue> struct AccTraits {
    using Host = THost;
    using Acc = TAcc;
    using Queue = TQueue;
    using DevHost = alpaka::dev::Dev<THost>;
    using DevAcc = alpaka::dev::Dev<TAcc>;
    using PltfHost = alpaka::pltf::Pltf<DevHost>;
    using PltfAcc = alpaka::pltf::Pltf<DevAcc>;
    using Event = alpaka::event::Event<TQueue>;
    template <typename T>
    using AccBuf = alpaka::mem::buf::Buf<DevAcc, T, Dim, Size>;
    template <typename T>
    using HostBuf = alpaka::mem::buf::Buf<DevHost, T, Dim, Size>;
    template <typename T>
    using HostView = alpaka::mem::view::ViewSubView<DevHost, T, Dim, Size>;
};

#ifdef ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED
//#############################################################################
//! OpenMP 2 Blocks defines
//!
//! Defines Host, Device, etc. for the OpenMP 2 Blocks accelerator.
struct CpuOmp2Blocks
    : public AccTraits<alpaka::acc::AccCpuOmp2Blocks<Dim, Size>,
                       alpaka::acc::AccCpuOmp2Blocks<Dim, Size>,
                       alpaka::queue::QueueCpuSync> {
    static constexpr std::size_t STREAMS_PER_DEV = 4;
    static constexpr Size elementsPerThread = 1u;
    static constexpr Size threadsPerBlock = 1u;
    static constexpr Size blocksPerGrid = MAPSIZE;
};

template <typename T, typename TBuf, typename... TArgs>
struct GetIterator<T, TBuf, alpaka::acc::AccCpuOmp2Blocks<TArgs...>> {
    using Iterator =
        IteratorCpu<alpaka::acc::AccCpuOmp2Blocks<TArgs...>, T, TBuf>;
};
#endif

#ifdef ALPAKA_ACC_CPU_BT_OMP4_ENABLED
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
//#############################################################################
//! OpenMP 4 defines
//!
//! Defines Host, Device, etc. for the OpenMP 4 accelerator.
struct CpuOmp4 : public AccTraits<alpaka::acc::AccCpuSerial<Dim, Size>,
                                  alpaka::acc::AccCpuOmp4<Dim, Size>,
                                  alpaka::queue::QueueCpuSync> {
    static constexpr std::size_t STREAMS_PER_DEV = 4;
    static constexpr Size elementsPerThread = 1u;
    static constexpr Size threadsPerBlock = 1u;
    static constexpr Size blocksPerGrid = MAPSIZE;
};

template <typename T, typename TBuf, typename... TArgs>
struct GetIterator<T, TBuf, alpaka::acc::AccCpuOmp4<TArgs...>> {
    using Iterator = IteratorCpu<alpaka::acc::AccCpuOmp4<TArgs...>, T, TBuf>;
};
#endif
#endif

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
//#############################################################################
//! Serial CPU defines
//!
//! Defines Host, Device, etc. for the serial CPU accelerator.
struct CpuSerial : public AccTraits<alpaka::acc::AccCpuSerial<Dim, Size>,
                                    alpaka::acc::AccCpuSerial<Dim, Size>,
                                    alpaka::queue::QueueCpuSync> {
    static constexpr std::size_t STREAMS_PER_DEV = 4;
    static constexpr Size elementsPerThread = 1u;
    static constexpr Size threadsPerBlock = 1u;
    static constexpr Size blocksPerGrid = MAPSIZE;
};

template <typename T, typename TBuf, typename... TArgs>
struct GetIterator<T, TBuf, alpaka::acc::AccCpuSerial<TArgs...>> {
    using Iterator = IteratorCpu<alpaka::acc::AccCpuSerial<TArgs...>, T, TBuf>;
};
#endif

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED
//#############################################################################
//! CPU Threads defines
//!
//! Defines Host, Device, etc. for the CPU Threads accelerator.
struct CpuThreads : public AccTraits<alpaka::acc::AccCpuThreads<Dim, Size>,
                                     alpaka::acc::AccCpuThreads<Dim, Size>,
                                     alpaka::queue::QueueCpuSync> {
    static constexpr std::size_t STREAMS_PER_DEV = 1;
    static constexpr Size elementsPerThread = 1u;
    static constexpr Size threadsPerBlock = 1u;
    static constexpr Size blocksPerGrid = MAPSIZE;
};

template <typename T, typename TBuf, typename... TArgs>
struct GetIterator<T, TBuf, alpaka::acc::AccCpuThreads<TArgs...>> {
    using Iterator = IteratorCpu<alpaka::acc::AccCpuThreads<TArgs...>, T, TBuf>;
};
#endif

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
//#############################################################################
//! CUDA defines
//!
//! Defines Host, Device, etc. for the CUDA accelerator.
struct GpuCudaRt : public AccTraits<alpaka::acc::AccCpuSerial<Dim, Size>,
                                    alpaka::acc::AccGpuCudaRt<Dim, Size>,
                                    alpaka::queue::QueueCudaRtAsync> {
    static constexpr std::size_t STREAMS_PER_DEV = 1;
    static constexpr Size elementsPerThread = 1u;
    static constexpr Size threadsPerBlock = 256;
    static constexpr Size blocksPerGrid = (MAPSIZE + 255) / 256;
};

template <typename T, typename TBuf, typename... TArgs>
struct GetIterator<T, TBuf, alpaka::acc::AccGpuCudaRt<TArgs...>> {
    using Iterator = IteratorGpu<alpaka::acc::AccGpuCudaRt<TArgs...>, T, TBuf>;
};
#endif
#endif
