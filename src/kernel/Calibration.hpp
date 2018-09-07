#pragma once
#include "../Config.hpp"
#include "helpers.hpp"

struct CalibrationKernel {
    template <typename TAcc, 
              typename TDetectorData, 
              typename TPedestalMap, 
              typename TMask,
              typename TNumFrames
             >
    ALPAKA_FN_ACC auto operator()(TAcc const& acc,
                                  TData const* const detectorData,
                                  TPedestal* const pedestalMap,
                                  TMask* const mask,
                                  TNumFrames const numFrames) {
        auto const globalThreadIdx =
            alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        auto const globalThreadExtent =
            alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

        auto const linearizedGlobalThreadIdx =
            alpaka::idx::mapIdx<1u>(globalThreadIdx, globalThreadExtent);

        auto id = linearizedGlobalThreadIdx[0u];
        const std::size_t FRAMESPERSTAGE[] = {
            FRAMESPERSTAGE_G0, FRAMESPERSTAGE_G1, FRAMESPERSTAGE_G2};

        // determine expected gain stage
        for (TNumFrames i = 0; i < numFrames; ++i) {
            // find expected gain stage
            char expectedGainStage;
            for (int i = 0; i < 3; ++i) {
                if (pedestalMap[i].count != FRAMESPERSTAGE[i]) {
                    expectedGainStage = i;
                    break;
                }
            }
            auto dataword = detectorData[i].data[id];
            auto adc = getAdc(dataword);
            auto gainStage = getGainStage(dataword);
            updatePedestal(acc, adc, pedestalMap[gainStage][id]);
            // mark pixel invalid if expected gainstage does not match
            if (expecedGainStage != gainStage) {
                mask[id] = false;
                }
            }
            ++expectedGainStage;
        }
    }
};
