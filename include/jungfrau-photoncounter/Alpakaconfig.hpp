/**
 * \file
 * Copyright 2019 Sebastian Benner, Jonas Schenke
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

#include "kernel/Iterator.hpp"
#include <alpaka/alpaka.hpp>

// Defines workdiv.
using WorkDiv = alpaka::WorkDivMembers<Dim, Size>;
using CpuSyncQueue = alpaka::QueueCpuBlocking;

template <typename TAccelerator> WorkDiv getWorkDiv() {
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
struct CpuFibers {};
#endif

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED
struct CpuOmp2Threads {};
#endif

#ifdef ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
//#############################################################################
//! Intel TBB defines
//!
//! Defines Host, Device, etc. for the Intel TBB accelerator.
template <std::size_t MAPSIZE> struct CpuTbbBlocks {
  using Host = alpaka::AccCpuTbbBlocks<Dim, Size>;
  using Acc = alpaka::AccCpuTbbBlocks<Dim, Size>;
  using Queue = alpaka::QueueCpuNonBlocking;
  using DevHost = alpaka::Dev<Host>;
  using DevAcc = alpaka::Dev<Acc>;
  using PltfHost = alpaka::Pltf<DevHost>;
  using PltfAcc = alpaka::Pltf<DevAcc>;
  using Event = alpaka::Event<Queue>;
  template <typename T>
  using AccBuf = alpaka::Buf<DevAcc, T, Dim, Size>;
  template <typename T>
  using HostBuf = alpaka::Buf<DevHost, T, Dim, Size>;
  template <typename T>
  using AccView = alpaka::ViewSubView<DevAcc, T, Dim, Size>;
  template <typename T>
  using HostView = alpaka::ViewSubView<DevHost, T, Dim, Size>;

  static constexpr std::size_t STREAMS_PER_DEV = 1u;
  static constexpr Size elementsPerThread = 8u;
  static constexpr Size threadsPerBlock = 1u;
  static constexpr Size blocksPerGrid =
      (MAPSIZE + elementsPerThread) / elementsPerThread;
};

template <typename T, typename TBuf, typename... TArgs>
struct GetIterator<T, TBuf, alpaka::AccCpuTbbBlocks<TArgs...>> {
  using Iterator = IteratorCpu<alpaka::AccCpuTbbBlocks<TArgs...>, T, TBuf>;
};
#endif
#endif

/*
template <typename THost, typename TAcc, typename TQueue> struct AccTraits {
    using Host = THost;
    using Acc = TAcc;
    using Queue = TQueue;
    using DevHost = alpaka::Dev<THost>;
    using DevAcc = alpaka::Dev<TAcc>;
    using PltfHost = alpaka::Pltf<DevHost>;
    using PltfAcc = alpaka::Pltf<DevAcc>;
    using Event = alpaka::Event<TQueue>;
    template <typename T>
    using AccBuf = alpaka::Buf<DevAcc, T, Dim, Size>;
    template <typename T>
    using HostBuf = alpaka::Buf<DevHost, T, Dim, Size>;
    template <typename T>
    using HostView = alpaka::ViewSubView<DevHost, T, Dim, Size>;
};
*/

#ifdef ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
//#############################################################################
//! OpenMP 2 Blocks defines
//!
//! Defines Host, Device, etc. for the OpenMP 2 Blocks accelerator.
template <std::size_t MAPSIZE> struct CpuOmp2Blocks {
  using Host = alpaka::AccCpuSerial<Dim, Size>;
  using Acc = alpaka::AccCpuOmp2Blocks<Dim, Size>;
  using Queue = alpaka::QueueCpuBlocking;
  using DevHost = alpaka::Dev<Host>;
  using DevAcc = alpaka::Dev<Acc>;
  using PltfHost = alpaka::Pltf<DevHost>;
  using PltfAcc = alpaka::Pltf<DevAcc>;
  using Event = alpaka::Event<Queue>;
  template <typename T>
  using AccBuf = alpaka::Buf<DevAcc, T, Dim, Size>;
  template <typename T>
  using HostBuf = alpaka::Buf<DevHost, T, Dim, Size>;
  template <typename T>
  using AccView = alpaka::ViewSubView<DevAcc, T, Dim, Size>;
  template <typename T>
  using HostView = alpaka::ViewSubView<DevHost, T, Dim, Size>;

  static constexpr std::size_t STREAMS_PER_DEV = 1;
  static constexpr Size elementsPerThread = 1920u;
  static constexpr Size threadsPerBlock = 1u;
  static constexpr Size blocksPerGrid =
      (MAPSIZE + elementsPerThread - 1) / elementsPerThread;
};

template <typename T, typename TBuf, typename... TArgs>
struct GetIterator<T, TBuf, alpaka::AccCpuOmp2Blocks<TArgs...>> {
  using Iterator =
      IteratorCpu<alpaka::AccCpuOmp2Blocks<TArgs...>, T, TBuf>;
};
#endif
#endif

#ifdef ALPAKA_ACC_CPU_BT_OMP4_ENABLED
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
//#############################################################################
//! OpenMP 4 defines
//!
//! Defines Host, Device, etc. for the OpenMP 4 accelerator.
template <std::size_t MAPSIZE> struct CpuOmp4 {
  using Host = alpaka::AccCpuOmp4<Dim, Size>;
  using Acc = alpaka::AccCpuOmp4<Dim, Size>;
  using Queue = alpaka::QueueCpuNonBlocking;
  using DevHost = alpaka::Dev<Host>;
  using DevAcc = alpaka::Dev<Acc>;
  using PltfHost = alpaka::Pltf<DevHost>;
  using PltfAcc = alpaka::Pltf<DevAcc>;
  using Event = alpaka::Event<Queue>;
  template <typename T>
  using AccBuf = alpaka::Buf<DevAcc, T, Dim, Size>;
  template <typename T>
  using HostBuf = alpaka::Buf<DevHost, T, Dim, Size>;
  template <typename T>
  using AccView = alpaka::ViewSubView<DevAcc, T, Dim, Size>;
  template <typename T>
  using HostView = alpaka::ViewSubView<DevHost, T, Dim, Size>;

  static constexpr std::size_t STREAMS_PER_DEV = 1u;
  static constexpr Size elementsPerThread = 8u;
  static constexpr Size threadsPerBlock = 1u;
  static constexpr Size blocksPerGrid =
      (MAPSIZE + elementsPerThread - 1) / elementsPerThread;
};

template <typename T, typename TBuf, typename... TArgs>
struct GetIterator<T, TBuf, alpaka::AccCpuOmp4<TArgs...>> {
  using Iterator = IteratorCpu<alpaka::AccCpuOmp4<TArgs...>, T, TBuf>;
};
#endif
#endif

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
//#############################################################################
//! Serial CPU defines
//!
//! Defines Host, Device, etc. for the serial CPU accelerator.
template <std::size_t MAPSIZE> struct CpuSerial {
  using Host = alpaka::AccCpuSerial<Dim, Size>;
  using Acc = alpaka::AccCpuSerial<Dim, Size>;
  using Queue = alpaka::QueueCpuBlocking;
  using DevHost = alpaka::Dev<Host>;
  using DevAcc = alpaka::Dev<Acc>;
  using PltfHost = alpaka::Pltf<DevHost>;
  using PltfAcc = alpaka::Pltf<DevAcc>;
  using Event = alpaka::Event<Queue>;
  template <typename T>
  using AccBuf = alpaka::Buf<DevAcc, T, Dim, Size>;
  template <typename T>
  using HostBuf = alpaka::Buf<DevHost, T, Dim, Size>;
  template <typename T>
  using AccView = alpaka::ViewSubView<DevAcc, T, Dim, Size>;
  template <typename T>
  using HostView = alpaka::ViewSubView<DevHost, T, Dim, Size>;

  static constexpr std::size_t STREAMS_PER_DEV = 1;
  static constexpr Size elementsPerThread = 1920u;
  static constexpr Size threadsPerBlock = 1u;
  static constexpr Size blocksPerGrid =
      (MAPSIZE + elementsPerThread - 1) / elementsPerThread;
};

template <typename T, typename TBuf, typename... TArgs>
struct GetIterator<T, TBuf, alpaka::AccCpuSerial<TArgs...>> {
  using Iterator = IteratorCpu<alpaka::AccCpuSerial<TArgs...>, T, TBuf>;
};
#endif

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED
//#############################################################################
//! CPU Threads defines
//!
//! Defines Host, Device, etc. for the CPU Threads accelerator.
template <std::size_t MAPSIZE> struct CpuThreads {
  using Host = alpaka::AccCpuThreads<Dim, Size>;
  using Acc = alpaka::AccCpuThreads<Dim, Size>;
  using Queue = alpaka::QueueCpuNonBlocking;
  using DevHost = alpaka::Dev<Host>;
  using DevAcc = alpaka::Dev<Acc>;
  using PltfHost = alpaka::Pltf<DevHost>;
  using PltfAcc = alpaka::Pltf<DevAcc>;
  using Event = alpaka::Event<Queue>;
  template <typename T>
  using AccBuf = alpaka::Buf<DevAcc, T, Dim, Size>;
  template <typename T>
  using HostBuf = alpaka::Buf<DevHost, T, Dim, Size>;
  template <typename T>
  using AccView = alpaka::ViewSubView<DevAcc, T, Dim, Size>;
  template <typename T>
  using HostView = alpaka::ViewSubView<DevHost, T, Dim, Size>;

  static constexpr std::size_t STREAMS_PER_DEV = 1;
  static constexpr Size elementsPerThread = 8u;
  static constexpr Size threadsPerBlock = 1u;
  static constexpr Size blocksPerGrid =
      (MAPSIZE + elementsPerThread - 1) / elementsPerThread;
};

template <typename T, typename TBuf, typename... TArgs>
struct GetIterator<T, TBuf, alpaka::AccCpuThreads<TArgs...>> {
  using Iterator = IteratorCpu<alpaka::AccCpuThreads<TArgs...>, T, TBuf>;
};
#endif
#endif

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
//#############################################################################
//! CUDA defines
//!
//! Defines Host, Device, etc. for the CUDA accelerator.
template <std::size_t MAPSIZE> struct GpuCudaRt {
  using Host = alpaka::AccCpuSerial<Dim, Size>;
  using Acc = alpaka::AccGpuCudaRt<Dim, Size>;
  using Queue = alpaka::QueueCudaRtNonBlocking;
  using DevHost = alpaka::Dev<Host>;
  using DevAcc = alpaka::Dev<Acc>;
  using PltfHost = alpaka::Pltf<DevHost>;
  using PltfAcc = alpaka::Pltf<DevAcc>;
  using Event = alpaka::Event<Queue>;
  template <typename T>
  using AccBuf = alpaka::Buf<DevAcc, T, Dim, Size>;
  template <typename T>
  using HostBuf = alpaka::Buf<DevHost, T, Dim, Size>;
  template <typename T>
  using AccView = alpaka::ViewSubView<DevAcc, T, Dim, Size>;
  template <typename T>
  using HostView = alpaka::ViewSubView<DevHost, T, Dim, Size>;

  static constexpr std::size_t STREAMS_PER_DEV = 3;
  static constexpr Size elementsPerThread = 1u;
  static constexpr Size threadsPerBlock = 256;
  static constexpr Size blocksPerGrid =
      (MAPSIZE + threadsPerBlock - 1) / threadsPerBlock;
};

template <typename T, typename TBuf, typename... TArgs>
struct GetIterator<T, TBuf, alpaka::AccGpuCudaRt<TArgs...>> {
  using Iterator = IteratorGpu<alpaka::AccGpuCudaRt<TArgs...>, T, TBuf>;
};

template <std::size_t MAPSIZE> struct GpuCudaRtSync {
  using Host = alpaka::AccCpuSerial<Dim, Size>;
  using Acc = alpaka::AccGpuCudaRt<Dim, Size>;
  using Queue = alpaka::QueueCudaRtBlocking;
  using DevHost = alpaka::Dev<Host>;
  using DevAcc = alpaka::Dev<Acc>;
  using PltfHost = alpaka::Pltf<DevHost>;
  using PltfAcc = alpaka::Pltf<DevAcc>;
  using Event = alpaka::Event<Queue>;
  template <typename T>
  using AccBuf = alpaka::Buf<DevAcc, T, Dim, Size>;
  template <typename T>
  using HostBuf = alpaka::Buf<DevHost, T, Dim, Size>;
  template <typename T>
  using AccView = alpaka::ViewSubView<DevAcc, T, Dim, Size>;
  template <typename T>
  using HostView = alpaka::ViewSubView<DevHost, T, Dim, Size>;

  static constexpr std::size_t STREAMS_PER_DEV = 1;
  static constexpr Size elementsPerThread = 1u;
  static constexpr Size threadsPerBlock = 256;
  static constexpr Size blocksPerGrid =
      (MAPSIZE + threadsPerBlock - 1) / threadsPerBlock;
};
#endif
#endif

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
//#############################################################################
//! HIP defines
//!
//! Defines Host, Device, etc. for the HIP accelerator.
template <std::size_t MAPSIZE> struct GpuHipRt {
  using Host = alpaka::acc::AccCpuSerial<Dim, Size>;
  using Acc = alpaka::acc::AccGpuHipRt<Dim, Size>;
  using Queue = alpaka::queue::QueueHipRtNonBlocking;
  using DevHost = alpaka::dev::Dev<Host>;
  using DevAcc = alpaka::dev::Dev<Acc>;
  using PltfHost = alpaka::pltf::Pltf<DevHost>;
  using PltfAcc = alpaka::pltf::Pltf<DevAcc>;
  using Event = alpaka::event::Event<Queue>;
  template <typename T>
  using AccBuf = alpaka::mem::buf::Buf<DevAcc, T, Dim, Size>;
  template <typename T>
  using HostBuf = alpaka::mem::buf::Buf<DevHost, T, Dim, Size>;
  template <typename T>
  using AccView = alpaka::mem::view::ViewSubView<DevAcc, T, Dim, Size>;
  template <typename T>
  using HostView = alpaka::mem::view::ViewSubView<DevHost, T, Dim, Size>;

  static constexpr std::size_t STREAMS_PER_DEV = 3;
  static constexpr Size elementsPerThread = 1u;
  static constexpr Size threadsPerBlock = 256;
  static constexpr Size blocksPerGrid =
      (MAPSIZE + threadsPerBlock - 1) / threadsPerBlock;
};

template <typename T, typename TBuf, typename... TArgs>
struct GetIterator<T, TBuf, alpaka::acc::AccGpuCudaRt<TArgs...>> {
  using Iterator = IteratorGpu<alpaka::acc::AccGpuCudaRt<TArgs...>, T, TBuf>;
};
#endif
#endif
