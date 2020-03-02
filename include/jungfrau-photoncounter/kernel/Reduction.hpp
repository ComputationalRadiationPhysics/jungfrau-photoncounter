#pragma once

#include "../AlpakaHelper.hpp"
#include "../CheapArray.hpp"
#include <alpaka/alpaka.hpp>

//#############################################################################
//! A reduction kernel.
//!
//! \tparam TBlockSize The block size.
//! \tparam T The data type.
//! \tparam TFunc The Functor type for the reduction function.
template <uint32_t TBlockSize, typename T> struct ReduceKernel {
    ALPAKA_NO_HOST_ACC_WARNING

    //-----------------------------------------------------------------------------
    //! The kernel entry point.
    //!
    //! \tparam TAcc The accelerator environment.
    //! \tparam TElem The element type.
    //! \tparam TIdx The index type.
    //!
    //! \param acc The accelerator object.
    //! \param source The source memory.
    //! \param destination The destination memory.
    //! \param n The problem size.
    //! \param func The reduction function.
    template <typename TAcc, typename TElem, typename TIdx>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc,
                                  TElem const* const source,
                                  TElem* destination,
                                  TIdx const& n) const -> void
    {
        auto& sdata(alpakaSharedMemory<CheapArray<T, TBlockSize>>(acc));

        const uint32_t blockIndex(
            static_cast<uint32_t>(alpakaGetBlockIdx(acc)[0]));
        const uint32_t threadIndex(
            static_cast<uint32_t>(alpakaGetThreadIdx(acc)[0]));
        const uint32_t gridDimension(
            static_cast<uint32_t>(alpakaGetGridDim(acc)[0]));

        // equivalent to blockIndex * TBlockSize + threadIndex
        const uint32_t linearizedIndex(
            static_cast<uint32_t>(getLinearIdx(acc)));

        typename GetIterator<T, T, TAcc>::Iterator it(
            acc, source->data, linearizedIndex, gridDimension * TBlockSize, n);

        auto func = [&](T a, T b) -> T { return alpakaMax(acc, a, b); };

        T result = 0; // suppresses compiler warnings

        if (threadIndex < n)
            result = *(it++); // avoids using the
                              // neutral element of specific

        // --------
        // Level 1: grid reduce, reading from global memory
        // --------

        // reduce per thread with increased ILP by 4x unrolling sum.
        // the thread of our block reduces its 4 grid-neighbored threads and
        // advances by grid-striding loop (maybe 128bit load improve perf)

        while (it + 3 < it.end()) {
            result = func(func(func(result, func(*it, *(it + 1))), *(it + 2)),
                          *(it + 3));
            it += 4;
        }

        // doing the remaining blocks
        while (it < it.end())
            result = func(result, *(it++));

        if (threadIndex < n)
            sdata[threadIndex] = result;

        alpakaSyncThreads(acc);

        // --------
        // Level 2: block + warp reduce, reading from shared memory
        // --------

        ALPAKA_UNROLL()
        for (uint32_t currentBlockSize = TBlockSize,
                      currentBlockSizeUp =
                          (TBlockSize + 1) / 2; // ceil(TBlockSize/2.0)
             currentBlockSize > 1;
             currentBlockSize = currentBlockSize / 2,
                      currentBlockSizeUp = (currentBlockSize + 1) /
                                           2) // ceil(currentBlockSize/2.0)
        {
            bool cond =
                threadIndex < currentBlockSizeUp // only first half of block
                                                 // is working
                && (threadIndex + currentBlockSizeUp) <
                       TBlockSize // index for second half must be in bounds
                && (blockIndex * TBlockSize + threadIndex +
                    currentBlockSizeUp) < n &&
                threadIndex <
                    n; // if elem in second half has been initialized before

            if (cond)
                sdata[threadIndex] =
                    func(sdata[threadIndex],
                         sdata[threadIndex + currentBlockSizeUp]);

            alpakaSyncThreads(acc);
        }

        // store block result to gmem
        if (threadIndex == 0 && threadIndex < n)
            destination->data[blockIndex] = sdata[0];
    }
};
