#pragma once
#include "../Config.hpp"
#include "helpers.hpp"

struct CalibrationKernel {
    template <typename TAcc, 
              typename TDetectorData, 
              typename TPedestalMap, 
              typename TNumFrames
             >
    ALPAKA_FN_ACC auto operator()(TAcc const& acc,
                                  TData const* const detectorData,
                                  TPedestal* const pedestalMap,
                                  TNumFrames const numFrames) const -> void
    {
        auto const globalThreadIdx =
            alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        auto const globalThreadExtent =
            alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

        auto const linearizedGlobalThreadIdx =
            alpaka::idx::mapIdx<1u>(globalThreadIdx, globalThreadExtent);

        auto id = linearizedGlobalThreadIdx[0u];
        for (TNumFrames i = 0; i < numFrames; ++i) {
            auto dataword = detectorData[i].data[id];
            auto adc = getAdc(dataword);
            auto gainStage = getGainStage(dataword);
            updatePedestal(acc, adc, pedestalMap[gainStage][id]);
        }
    }
};
