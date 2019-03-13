#pragma once
#include "../Config.hpp"
#include "helpers.hpp"

struct GainStageMaskingKernel {
    template <typename TAcc,
              typename TGainStageMap,
              typename TNumFrames,
              typename TMask>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc,
                                  TGainStageMap* const inputGainStageMaps,
                                  TGainStageMap* outputGainStageMaps,
                                  TNumFrames const numFrame,
                                  TMask const* const mask) const -> void
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
        
        // use masks to check whether the channel is valid or masked out
        bool isValid = !mask ? 1 : mask->data[id];

        outputGainStageMaps->data[id] =
            (isValid ? inputGainStageMaps[numFrame].data[id] : MASKED_VALUE);
    }
};
