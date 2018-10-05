#pragma once
#include "../Config.hpp"
#include "helpers.hpp"

struct ConversionKernel {
    template <typename TAcc,
              typename TDetectorData,
              typename TGainMap,
              typename TPedestalMap,
              typename TGainStageMap,
              typename TEnergyMap,
              typename TNumFrames,
              typename TMask>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc,
                                  TDetectorData const* const detectorData,
                                  TGainMap const* const gainMaps,
                                  TPedestalMap* const pedestalMaps,
                                  TGainStageMap* const gainStageMaps,
                                  TEnergyMap* const energyMaps,
                                  TNumFrames const numFrames,
                                  TMask const* const mask) const -> void
    {
        auto const globalThreadIdx =
            alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        auto const globalThreadExtent =
            alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

        auto const linearizedGlobalThreadIdx =
            alpaka::idx::mapIdx<1u>(globalThreadIdx, globalThreadExtent);

        auto id = linearizedGlobalThreadIdx[0u];

        // use masks to check whether the channel is valid or masked out
        bool isValid = mask->data[id];

        for (TNumFrames i = 0; i < numFrames; ++i) {
            processInput(acc, 
                         detectorData[i], 
                         gainMaps, 
                         pedestalMaps, 
                         gainStageMaps[i],
                         energyMaps[i],
                         mask,
                         id);            
        }
    }
};
