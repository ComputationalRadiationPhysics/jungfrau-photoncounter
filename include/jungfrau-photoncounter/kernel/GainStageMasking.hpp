#pragma once
#include "helpers.hpp"

template<typename Config>
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
        auto id = getLinearIdx(acc);
        
        // check range
        if (id >= Config::MAPSIZE)
            return;
        
        // use masks to check whether the channel is valid or masked out
        bool isValid = !mask ? 1 : mask->data[id];

        outputGainStageMaps->data[id] =
            (isValid ? inputGainStageMaps[numFrame].data[id] : Config::MASKED_VALUE);
    }
};
