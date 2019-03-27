#pragma once

#include <alpaka/alpaka.hpp>

// Defines types and size.
using Dim = alpaka::dim::DimInt<1u>;
using Size = uint64_t;
using Vec = alpaka::vec::Vec<Dim, Size>;

/**
 * Host Functions.
 **/

// rename alpaka alloc
template <typename T, typename... TArgs>
static inline auto alpakaAlloc(TArgs&&... args)
    -> decltype(alpaka::mem::buf::alloc<T, Size>(std::forward<TArgs>(args)...))
{
    // allocate buffer
    auto res = alpaka::mem::buf::alloc<T, Size>(std::forward<TArgs>(args)...);

    // pin memory
    alpaka::mem::buf::prepareForAsyncCopy(res);
    return res;
}

// rename alpaka getPtrNative
template <typename... TArgs>
static inline auto alpakaNativePtr(TArgs&&... args)
    -> decltype(alpaka::mem::view::getPtrNative(std::forward<TArgs>(args)...))
{
    return alpaka::mem::view::getPtrNative(std::forward<TArgs>(args)...);
}

// function to get first host device
template <typename TAlpaka>
static inline auto alpakaGetHost()
    -> decltype(alpaka::pltf::getDevByIdx<typename TAlpaka::PltfHost>(0u))
{
    return alpaka::pltf::getDevByIdx<typename TAlpaka::PltfHost>(0u);
}

// rename alpaka copy
template <typename... TArgs>
static inline auto alpakaCopy(TArgs&&... args)
    -> decltype(alpaka::mem::view::copy(std::forward<TArgs>(args)...))
{
    return alpaka::mem::view::copy(std::forward<TArgs>(args)...);
}

// rename alpaka enqueue kernel
template <typename... TArgs>
static inline auto alpakaEnqueueKernel(TArgs&&... args)
    -> decltype(alpaka::queue::enqueue(std::forward<TArgs>(args)...))
{
    return alpaka::queue::enqueue(std::forward<TArgs>(args)...);
}

// rename alpaka wait
template <typename... TArgs>
static inline auto alpakaWait(TArgs&&... args)
    -> decltype(alpaka::wait::wait(std::forward<TArgs>(args)...))
{
    return alpaka::wait::wait(std::forward<TArgs>(args)...);
}

// rename alpaka create kernel
template <typename TAlpaka, typename... TArgs>
static inline auto alpakaCreateKernel(TArgs&&... args)
    -> decltype(alpaka::kernel::createTaskKernel<typename TAlpaka::Acc>(
        std::forward<TArgs>(args)...))
{
    return alpaka::kernel::createTaskKernel<typename TAlpaka::Acc>(
        std::forward<TArgs>(args)...);
}

// rename alpaka set
template <typename... TArgs>
static inline auto alpakaMemSet(TArgs&&... args)
    -> decltype(alpaka::mem::view::set(std::forward<TArgs>(args)...))
{
    return alpaka::mem::view::set(std::forward<TArgs>(args)...);
}

// rename alpaka view plain pointer
template <typename TAlpaka, typename TData, typename... TArgs>
static inline auto alpakaViewPlainPtrHost(TArgs&&... args)
    -> decltype(alpaka::mem::view::
                    ViewPlainPtr<typename TAlpaka::DevHost, TData, Dim, Size>(
                        std::forward<TArgs>(args)...))
{
    return alpaka::mem::view::
        ViewPlainPtr<typename TAlpaka::DevHost, TData, Dim, Size>(
            std::forward<TArgs>(args)...);
}

// rename alpaka get dev count
template <typename TAlpaka, typename... TArgs>
static inline auto alpakaGetDevCount(TArgs&&... args)
    -> decltype(alpaka::pltf::getDevCount<typename TAlpaka::PltfAcc>(
        std::forward<TArgs>(args)...))
{
    return alpaka::pltf::getDevCount<typename TAlpaka::PltfAcc>(
        std::forward<TArgs>(args)...);
}

// rename alpaka get devs
template <typename TAlpaka, typename... TArgs>
static inline auto alpakaGetDevs(TArgs&&... args)
    -> decltype(alpaka::pltf::getDevs<typename TAlpaka::PltfAcc>(
        std::forward<TArgs>(args)...))
{
    return alpaka::pltf::getDevs<typename TAlpaka::PltfAcc>(
        std::forward<TArgs>(args)...);
}

// rename alpaka getMemBytes
template <typename... TArgs>
static inline auto alpakaGetMemBytes(TArgs&&... args)
    -> decltype(alpaka::dev::getMemBytes(std::forward<TArgs>(args)...))
{
    return alpaka::dev::getMemBytes(std::forward<TArgs>(args)...);
}

// rename alpaka getFreeMemBytes
template <typename... TArgs>
static inline auto alpakaGetFreeMemBytes(TArgs&&... args)
    -> decltype(alpaka::dev::getFreeMemBytes(std::forward<TArgs>(args)...))
{
    return alpaka::dev::getFreeMemBytes(std::forward<TArgs>(args)...);
}

/**
 * Device Functions.
 **/

// rename alpaka max
template <typename... TArgs>
ALPAKA_FN_ACC ALPAKA_FN_INLINE static auto alpakaMax(TArgs&&... args)
    -> decltype(alpaka::math::max(std::forward<TArgs>(args)...))
{
    return alpaka::math::max(std::forward<TArgs>(args)...);
}

// rename alpaka min
template <typename... TArgs>
ALPAKA_FN_ACC ALPAKA_FN_INLINE static auto alpakaMin(TArgs&&... args)
    -> decltype(alpaka::math::min(std::forward<TArgs>(args)...))
{
    return alpaka::math::min(std::forward<TArgs>(args)...);
}

// rename alpaka sqrt
template <typename... TArgs>
ALPAKA_FN_ACC ALPAKA_FN_INLINE static auto alpakaSqrt(TArgs&&... args)
    -> decltype(alpaka::math::sqrt(std::forward<TArgs>(args)...))
{
    return alpaka::math::sqrt(std::forward<TArgs>(args)...);
}

// rename alpaka atomic add
template <typename... TArgs>
ALPAKA_FN_ACC ALPAKA_FN_INLINE static auto alpakaAtomicAdd(TArgs&&... args)
    -> decltype(alpaka::atomic::atomicOp<alpaka::atomic::op::Add>(
        std::forward<TArgs>(args)...))
{
    return alpaka::atomic::atomicOp<alpaka::atomic::op::Add>(
        std::forward<TArgs>(args)...);
}

// rename alpaka shared memory
template <typename TData, typename... TArgs>
ALPAKA_FN_ACC ALPAKA_FN_INLINE static auto alpakaSharedMemory(TArgs&&... args)
    -> decltype(alpaka::block::shared::st::allocVar<TData, __COUNTER__>(
        std::forward<TArgs>(args)...))
{
    return alpaka::block::shared::st::allocVar<TData, __COUNTER__>(
        std::forward<TArgs>(args)...);
}

// rename alpaka get global thread idx
template <typename... TArgs>
ALPAKA_FN_ACC ALPAKA_FN_INLINE static auto
alpakaGetGlobalThreadIdx(TArgs&&... args)
    -> decltype(alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(
        std::forward<TArgs>(args)...))
{
    return alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(
        std::forward<TArgs>(args)...);
}

// rename alpaka get global thread extent
template <typename... TArgs>
ALPAKA_FN_ACC ALPAKA_FN_INLINE static auto
alpakaGetGlobalThreadExtent(TArgs&&... args)
    -> decltype(alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Threads>(
        std::forward<TArgs>(args)...))
{
    return alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Threads>(
        std::forward<TArgs>(args)...);
}

// rename alpaka get linearized global thread idx
template <typename... TArgs>
ALPAKA_FN_ACC ALPAKA_FN_INLINE static auto
alpakaGetGlobalLinearizedGlobalThreadIdx(TArgs&&... args)
    -> decltype(alpaka::idx::mapIdx<1u>(std::forward<TArgs>(args)...))
{
    return alpaka::idx::mapIdx<1u>(std::forward<TArgs>(args)...);
}

// rename alpaka sync threads
template <typename... TArgs>
ALPAKA_FN_ACC ALPAKA_FN_INLINE static auto alpakaSyncThreads(TArgs&&... args)
    -> decltype(
        alpaka::block::sync::syncBlockThreads(std::forward<TArgs>(args)...))
{
    return alpaka::block::sync::syncBlockThreads(std::forward<TArgs>(args)...);
}

// get block index
template <typename... TArgs>
ALPAKA_FN_ACC ALPAKA_FN_INLINE static auto
alpakaGetBlockIdx(TArgs&&... args)
    -> decltype(alpaka::idx::getIdx<alpaka::Grid, alpaka::Blocks>(std::forward<TArgs>(args)...))
{
    return alpaka::idx::getIdx<alpaka::Grid, alpaka::Blocks>(std::forward<TArgs>(args)...);
}

// get thread index
template <typename... TArgs>
ALPAKA_FN_ACC ALPAKA_FN_INLINE static auto
alpakaGetThreadIdx(TArgs&&... args)
    -> decltype(alpaka::idx::getIdx<alpaka::Block, alpaka::Threads>(std::forward<TArgs>(args)...))
{
    return alpaka::idx::getIdx<alpaka::Block, alpaka::Threads>(std::forward<TArgs>(args)...);
}

// get grid dimension
template <typename... TArgs>
ALPAKA_FN_ACC ALPAKA_FN_INLINE static auto
alpakaGetGridDim(TArgs&&... args)
    -> decltype(alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Blocks>(std::forward<TArgs>(args)...))
{
    return alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Blocks>(std::forward<TArgs>(args)...);
}
