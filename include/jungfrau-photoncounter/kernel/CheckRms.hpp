#pragma once
#include "../Config.hpp"
#include "helpers.hpp"

struct CheckRmsKernel {
    template <typename TAcc,
              typename TInitPedestalMap,
              typename TMaskMap,
              typename TRmsThreshold>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc,
                                  TInitPedestalMap const* const initPedestalMap,
                                  TMaskMap* const mask,
                                  TRmsThreshold const threshold) const -> void
    {
        auto const globalThreadIdx =
            alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        auto const globalThreadExtent =
            alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

        auto const linearizedGlobalThreadIdx =
            alpaka::idx::mapIdx<1u>(globalThreadIdx, globalThreadExtent);

        auto id = linearizedGlobalThreadIdx[0u];

        // check range
        if (id >= MAPSIZE)
            return;

        // check if measured RMS exceeds threshold
        if (initPedestalMap[0][id].sumSquares >
            threshold * threshold * MOVING_STAT_WINDOW_SIZE)
            mask->data[id] = false;
    }
};
