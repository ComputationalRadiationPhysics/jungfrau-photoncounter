#pragma once

#include "../Config.hpp"

struct GainmapInversionKernel {
    template <typename TAcc, typename TGain>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc,
                                  TGain* const gainmaps) 
        const -> void
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
        
        for (size_t i = 0; i < GAINMAPS; ++i) {
            gainmaps[i][id] = 1.0 / gainmaps[i][id];
        }
    }
};

