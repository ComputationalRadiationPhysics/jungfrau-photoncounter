#pragma once
#include "../Config.hpp"
#include "helpers.hpp"

struct CalibrationKernel {
    template <typename TAcc,
              typename TDetectorData,
              typename TInitPedestalMap,
              typename TPedestalMap,
              typename TMaskMap,
              typename TNumFrames>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc,
                                  TDetectorData const* const detectorData,
                                  TInitPedestalMap* const initPedestalMap,
                                  TPedestalMap* const pedestalMap,
                                  TMaskMap* const mask,
                                  TNumFrames const numFrames) const -> void
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

        const std::size_t FRAMESPERSTAGE[] = {
            FRAMESPERSTAGE_G0, FRAMESPERSTAGE_G1, FRAMESPERSTAGE_G2};

        // find expected gain stage
        char expectedGainStage;
        for (int i = 0; i < PEDEMAPS; ++i) {
            if (initPedestalMap[i][id].count != FRAMESPERSTAGE[i]) {
                expectedGainStage = i;
                break;
            }
        }

        // determine expected gain stage
        for (TNumFrames i = 0; i < numFrames; ++i) {
            if (initPedestalMap[expectedGainStage][id].count ==
                FRAMESPERSTAGE[expectedGainStage]) {
                ++expectedGainStage;
            }

            auto dataword = detectorData[i].data[id];
            auto adc = getAdc(dataword);
            uint8_t gainStage = getGainStage(dataword);

            if (initPedestalMap[expectedGainStage][id].count <
                MOVING_STAT_WINDOW_SIZE) {
                initPedestal(acc, adc, initPedestalMap[gainStage][id]);
            }
            else {
                // manually increment counter
                ++initPedestalMap[expectedGainStage][id].count;
                updatePedestal(adc,
                               MOVING_STAT_WINDOW_SIZE,
                               pedestalMap[expectedGainStage][id]);
            }

            // copy readily calculated pedestal values into output
            // pedestal map
            pedestalMap[expectedGainStage][id] =
                initPedestalMap[expectedGainStage][id].mean;

            // mark pixel invalid if expected gainstage does not match
            if (expectedGainStage != gainStage) {
                mask->data[id] = false;
            }
        }
    }
};
