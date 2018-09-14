#pragma once
#include "../Config.hpp"
#include "helpers.hpp"

struct CalibrationKernel {
    template <typename TAcc,
              typename TDetectorData,
              typename TPedestalMap,
              typename TMaskMap,
              typename TNumFrames>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc,
                                  TDetectorData const* const detectorData,
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
        const std::size_t FRAMESPERSTAGE[] = {
            FRAMESPERSTAGE_G0, FRAMESPERSTAGE_G1, FRAMESPERSTAGE_G2};
        
        // find expected gain stage
        char expectedGainStage;
        for (int i = 0; i < PEDEMAPS; ++i) {
            if (pedestalMap[i][id].count != FRAMESPERSTAGE[i]) {
                expectedGainStage = i;
                break;
            }
        }

        // determine expected gain stage
        for (TNumFrames i = 0; i < numFrames; ++i) {
          if (pedestalMap[expectedGainStage][id].count == FRAMESPERSTAGE[i])
                ++expectedGainStage;
            auto dataword = detectorData[i].data[id];
          /*  auto adc = getAdc(dataword);
            auto gainStage = getGainStage(dataword);
            updatePedestal(acc, adc, pedestalMap[gainStage][id]);*/
            // mark pixel invalid if expected gainstage does not match
            //if (expectedGainStage != gainStage) {
              if(!mask)
                printf("REEEEEEEEEEEE\n");
              else
                mask->data[0] = false;
              //}
        }
    }
};
