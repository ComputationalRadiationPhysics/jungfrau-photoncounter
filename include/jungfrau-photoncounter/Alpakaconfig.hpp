#pragma once

#include "AlpakaHelper.hpp"
#include <alpaka/alpaka.hpp>

// Defines workdiv.
using WorkDiv = alpaka::workdiv::WorkDivMembers<Dim, Size>;
using CpuSyncQueue = alpaka::queue::QueueCpuBlocking;

template <typename TAccelerator> WorkDiv getWorkDiv() {
  return WorkDiv(
      decltype(TAccelerator::blocksPerGrid)(TAccelerator::blocksPerGrid),
      decltype(TAccelerator::threadsPerBlock)(TAccelerator::threadsPerBlock),
      decltype(TAccelerator::elementsPerThread)(
          TAccelerator::elementsPerThread));
}
  
#ifdef ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
//#############################################################################
//! OpenMP 2 Blocks defines
//!
//! Defines Host, Device, etc. for the OpenMP 2 Blocks accelerator.
template <std::size_t MAPSIZE> struct CpuOmp2Blocks {
  using Host = alpaka::acc::AccCpuSerial<Dim, Size>;
  using Acc = alpaka::acc::AccCpuOmp2Blocks<Dim, Size>;
  using Queue = alpaka::queue::QueueCpuBlocking;
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

  static constexpr std::size_t STREAMS_PER_DEV = 1;
  static constexpr Size elementsPerThread = 2u;//256u;
  static constexpr Size threadsPerBlock = 1u;
  static constexpr Size blocksPerGrid = MAPSIZE / 2;//(MAPSIZE + 255) / 256;
};
#endif
#endif


#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
//#############################################################################
//! Serial CPU defines
//!
//! Defines Host, Device, etc. for the serial CPU accelerator.
template <std::size_t MAPSIZE> struct CpuSerial {
  using Host = alpaka::acc::AccCpuSerial<Dim, Size>;
  using Acc = alpaka::acc::AccCpuSerial<Dim, Size>;
  using Queue = alpaka::queue::QueueCpuBlocking;
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

  static constexpr std::size_t STREAMS_PER_DEV = 1;
  static constexpr Size elementsPerThread = 256u;
  static constexpr Size threadsPerBlock = 1u;
  static constexpr Size blocksPerGrid = (MAPSIZE + 255) / 256;
};
#endif

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
//#############################################################################
//! CUDA defines
//!
//! Defines Host, Device, etc. for the CUDA accelerator.
template <std::size_t MAPSIZE> struct GpuCudaRt {
  using Host = alpaka::acc::AccCpuSerial<Dim, Size>;
  using Acc = alpaka::acc::AccGpuCudaRt<Dim, Size>;
  using Queue = alpaka::queue::QueueCudaRtNonBlocking;
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
  //! @note: only one element per thread allowed for CUDA (see issue #66)
  static constexpr Size elementsPerThread = 1u;
  static constexpr Size threadsPerBlock = 256;
  static constexpr Size blocksPerGrid = (MAPSIZE + 255) / 256;
};

template <std::size_t MAPSIZE> struct GpuCudaRtSync {
  using Host = alpaka::acc::AccCpuSerial<Dim, Size>;
  using Acc = alpaka::acc::AccGpuCudaRt<Dim, Size>;
  using Queue = alpaka::queue::QueueCudaRtBlocking;
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

  static constexpr std::size_t STREAMS_PER_DEV = 1;
  //! @note: only one element per thread allowed for CUDA (see issue #66)
  static constexpr Size elementsPerThread = 1u;
  static constexpr Size threadsPerBlock = 256;
  static constexpr Size blocksPerGrid = (MAPSIZE + 255) / 256;
};
#endif
#endif
