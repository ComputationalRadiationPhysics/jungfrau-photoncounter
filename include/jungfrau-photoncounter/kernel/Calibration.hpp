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
        auto id = getLinearIdx(acc);

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

            if (expectedGainStage == 0) {
                // select moving window for gain stage 0
                initPedestal(acc,
                             adc,
                             initPedestalMap[gainStage][id],
                             MOVING_STAT_WINDOW_SIZE);
            }
            else {
                // set moving window size for other pedestal stages to the
                // number of images available which effectively disables the
                // moving window
                initPedestal(acc,
                             adc,
                             initPedestalMap[gainStage][id],
                             FRAMESPERSTAGE[gainStage]);
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
