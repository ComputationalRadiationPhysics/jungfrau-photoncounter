#pragma once

#include <alpaka/alpaka.hpp>
#define BOOST_OPTIONAL_USE_OLD_DEFINITION_OF_NONE
#include <boost/optional.hpp>

// Defines types and size.
using Dim = alpaka::DimInt<1u>;
using Size = uint64_t;
using Vec = alpaka::Vec<Dim, Size>;

/**
 * Host Functions.
 **/

// rename alpaka alloc
template <typename T, typename... TArgs>
static inline auto alpakaAlloc(TArgs &&...args)
    -> decltype(alpaka::allocBuf<T, Size>(std::forward<TArgs>(args)...)) {
  // allocate buffer
  auto res = alpaka::allocBuf<T, Size>(std::forward<TArgs>(args)...);

  // pin memory
  alpaka::prepareForAsyncCopy(res);
  return res;
}

// rename alpaka getPtrNative
template <typename... TArgs>
static inline auto alpakaNativePtr(TArgs &&...args)
    -> decltype(alpaka::getPtrNative(std::forward<TArgs>(args)...)) {
  return alpaka::getPtrNative(std::forward<TArgs>(args)...);
}

// rename alpaka copy
template <typename... TArgs>
static inline auto alpakaCopy(TArgs &&...args)
    -> decltype(alpaka::memcpy(std::forward<TArgs>(args)...)) {
  return alpaka::memcpy(std::forward<TArgs>(args)...);
}

// rename alpaka enqueue kernel
template <typename... TArgs>
static inline auto alpakaEnqueueKernel(TArgs &&...args)
    -> decltype(alpaka::enqueue(std::forward<TArgs>(args)...)) {
  return alpaka::enqueue(std::forward<TArgs>(args)...);
}

// rename alpaka wait
template <typename... TArgs>
static inline auto alpakaWait(TArgs &&...args)
    -> decltype(alpaka::wait(std::forward<TArgs>(args)...)) {
  return alpaka::wait(std::forward<TArgs>(args)...);
}

// rename alpaka create kernel
template <typename TAlpaka, typename... TArgs>
static inline auto alpakaCreateKernel(TArgs &&...args)
    -> decltype(alpaka::createTaskKernel<typename TAlpaka::Acc>(
        std::forward<TArgs>(args)...)) {
  return alpaka::createTaskKernel<typename TAlpaka::Acc>(
      std::forward<TArgs>(args)...);
}

// rename alpaka set
template <typename... TArgs>
static inline auto alpakaMemSet(TArgs &&...args)
    -> decltype(alpaka::memset(std::forward<TArgs>(args)...)) {
  return alpaka::memset(std::forward<TArgs>(args)...);
}

// rename alpaka view plain pointer
template <typename TAlpaka, typename TData, typename... TArgs>
static inline auto alpakaViewPlainPtrHost(TArgs &&...args)
    -> decltype(alpaka::ViewPlainPtr<typename TAlpaka::DevHost, TData, Dim,
                                     Size>(std::forward<TArgs>(args)...)) {
  return alpaka::ViewPlainPtr<typename TAlpaka::DevHost, TData, Dim, Size>(
      std::forward<TArgs>(args)...);
}

// function to get accelerator device by ID
template <typename TAlpaka>
static inline auto alpakaGetDevByIdx(std::size_t idx)
    -> decltype(alpaka::getDevByIdx<typename TAlpaka::PltfAcc>(idx)) {
  return alpaka::getDevByIdx<typename TAlpaka::PltfAcc>(idx);
}

// function to get first host device
template <typename TAlpaka>
static inline auto alpakaGetHost()
    -> decltype(alpaka::getDevByIdx<typename TAlpaka::PltfHost>(0u)) {
  return alpaka::getDevByIdx<typename TAlpaka::PltfHost>(0u);
}

// rename alpaka get dev count
template <typename TAlpaka, typename... TArgs>
static inline auto alpakaGetDevCount(TArgs &&...args)
    -> decltype(alpaka::getDevCount<typename TAlpaka::PltfAcc>(
        std::forward<TArgs>(args)...)) {
  return alpaka::getDevCount<typename TAlpaka::PltfAcc>(
      std::forward<TArgs>(args)...);
}

// rename alpaka get devs
template <typename TAlpaka, typename... TArgs>
static inline auto alpakaGetDevs(TArgs &&...args) -> decltype(
    alpaka::getDevs<typename TAlpaka::PltfAcc>(std::forward<TArgs>(args)...)) {
  return alpaka::getDevs<typename TAlpaka::PltfAcc>(
      std::forward<TArgs>(args)...);
}

// rename alpaka getMemBytes
template <typename... TArgs>
static inline auto alpakaGetMemBytes(TArgs &&...args)
    -> decltype(alpaka::getMemBytes(std::forward<TArgs>(args)...)) {
  return alpaka::getMemBytes(std::forward<TArgs>(args)...);
}

// rename alpaka getFreeMemBytes
template <typename... TArgs>
static inline auto alpakaGetFreeMemBytes(TArgs &&...args)
    -> decltype(alpaka::getFreeMemBytes(std::forward<TArgs>(args)...)) {
  return alpaka::getFreeMemBytes(std::forward<TArgs>(args)...);
}

/**
 * Device Functions.
 **/

// rename alpaka max
template <typename... TArgs>
ALPAKA_FN_ACC ALPAKA_FN_INLINE static auto alpakaMax(TArgs &&...args)
    -> decltype(alpaka::math::max(std::forward<TArgs>(args)...)) {
  return alpaka::math::max(std::forward<TArgs>(args)...);
}

// rename alpaka min
template <typename... TArgs>
ALPAKA_FN_ACC ALPAKA_FN_INLINE static auto alpakaMin(TArgs &&...args)
    -> decltype(alpaka::math::min(std::forward<TArgs>(args)...)) {
  return alpaka::math::min(std::forward<TArgs>(args)...);
}

// rename alpaka sqrt
template <typename... TArgs>
ALPAKA_FN_ACC ALPAKA_FN_INLINE static auto alpakaSqrt(TArgs &&...args)
    -> decltype(alpaka::math::sqrt(std::forward<TArgs>(args)...)) {
  return alpaka::math::sqrt(std::forward<TArgs>(args)...);
}

// rename alpaka getExtent
template <std::size_t Tidx, typename TExtent>
ALPAKA_FN_ACC ALPAKA_FN_INLINE static auto
alpakaGetExtent(TExtent const &extent = TExtent())
    -> decltype(alpaka::extent::getExtent<Tidx, TExtent>(extent)) {
  return alpaka::extent::getExtent<Tidx, TExtent>(extent);
}

// rename alpaka atomic add
template <typename... TArgs>
ALPAKA_FN_ACC ALPAKA_FN_INLINE static auto alpakaAtomicAdd(TArgs &&...args)
    -> decltype(alpaka::atomicAdd(
        std::forward<TArgs>(args)..., alpaka::hierarchy::Blocks{})) {
  return alpaka::atomicAdd(std::forward<TArgs>(args)...,
                                             alpaka::hierarchy::Blocks{});
}

// rename alpaka shared memory
template <typename TData, typename... TArgs>
ALPAKA_FN_ACC ALPAKA_FN_INLINE static auto alpakaSharedMemory(TArgs &&...args)
    -> decltype(
        alpaka::allocVar<TData, __COUNTER__>(std::forward<TArgs>(args)...)) {
  return alpaka::allocVar<TData, __COUNTER__>(std::forward<TArgs>(args)...);
}

// rename alpaka get global thread idx
template <typename... TArgs>
ALPAKA_FN_ACC ALPAKA_FN_INLINE static auto
alpakaGetGlobalThreadIdx(TArgs &&...args)
    -> decltype(alpaka::getIdx<alpaka::Grid, alpaka::Threads>(
        std::forward<TArgs>(args)...)) {
  return alpaka::getIdx<alpaka::Grid, alpaka::Threads>(
      std::forward<TArgs>(args)...);
}

// rename alpaka get global thread extent
template <typename... TArgs>
ALPAKA_FN_ACC ALPAKA_FN_INLINE static auto
alpakaGetGlobalThreadExtent(TArgs &&...args)
    -> decltype(alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(
        std::forward<TArgs>(args)...)) {
  return alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(
      std::forward<TArgs>(args)...);
}

// rename alpaka get element extent
template <typename... TArgs>
ALPAKA_FN_ACC ALPAKA_FN_INLINE static auto
alpakaGetElementExtent(TArgs &&...args)
    -> decltype(alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(
        std::forward<TArgs>(args)...)) {
  return alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(
      std::forward<TArgs>(args)...);
}

// rename alpaka get linearized global thread idx
template <typename... TArgs>
ALPAKA_FN_ACC ALPAKA_FN_INLINE static auto
alpakaGetGlobalLinearizedGlobalThreadIdx(TArgs &&...args)
    -> decltype(alpaka::mapIdx<1u>(std::forward<TArgs>(args)...)) {
  return alpaka::mapIdx<1u>(std::forward<TArgs>(args)...);
}

// rename alpaka sync threads
template <typename... TArgs>
ALPAKA_FN_ACC ALPAKA_FN_INLINE static auto alpakaSyncThreads(TArgs &&...args)
    -> decltype(alpaka::syncBlockThreads(std::forward<TArgs>(args)...)) {
  return alpaka::syncBlockThreads(std::forward<TArgs>(args)...);
}

// get block index
template <typename... TArgs>
ALPAKA_FN_ACC ALPAKA_FN_INLINE static auto alpakaGetBlockIdx(TArgs &&...args)
    -> decltype(alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(
        std::forward<TArgs>(args)...)) {
  return alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(
      std::forward<TArgs>(args)...);
}

// get thread index
template <typename... TArgs>
ALPAKA_FN_ACC ALPAKA_FN_INLINE static auto alpakaGetThreadIdx(TArgs &&...args)
    -> decltype(alpaka::getIdx<alpaka::Block, alpaka::Threads>(
        std::forward<TArgs>(args)...)) {
  return alpaka::getIdx<alpaka::Block, alpaka::Threads>(
      std::forward<TArgs>(args)...);
}

// get grid dimension
template <typename... TArgs>
ALPAKA_FN_ACC ALPAKA_FN_INLINE static auto alpakaGetGridDim(TArgs &&...args)
    -> decltype(alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(
        std::forward<TArgs>(args)...)) {
  return alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(
      std::forward<TArgs>(args)...);
}
