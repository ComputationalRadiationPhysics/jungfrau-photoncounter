#pragma once
#include "helpers.hpp"

template <typename Config> struct CalibrationKernel {
  template <typename TAcc, typename TDetectorData, typename TInitPedestalMap,
            typename TPedestalMap, typename TMaskMap, typename TNumFrames>
  ALPAKA_FN_ACC auto
  operator()(TAcc const &acc, TDetectorData const *const detectorData,
             TInitPedestalMap *const initPedestalMap,
             TPedestalMap *const pedestalMap, TMaskMap *const mask,
             TNumFrames const numFrames) const -> void {
    constexpr auto PEDEMAPS = Config::PEDEMAPS;

    auto globalId = getLinearIdx(acc);
    auto elementsPerThread = getLinearElementExtent(acc);

    // iterate over all elements in the thread
    for (auto id = globalId * elementsPerThread;
         id < (globalId + 1) * elementsPerThread; ++id) {
      // check range
      if (id >= Config::MAPSIZE)
        break;

      const std::size_t FRAMESPERSTAGE[] = {Config::FRAMESPERSTAGE_G0,
                                            Config::FRAMESPERSTAGE_G1,
                                            Config::FRAMESPERSTAGE_G2};

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
          initPedestal(acc, adc, initPedestalMap[gainStage][id],
                       Config::MOVING_STAT_WINDOW_SIZE);
        } else {
          // set moving window size for other pedestal stages to the
          // number of images available which effectively disables the
          // moving window
          initPedestal(acc, adc, initPedestalMap[gainStage][id],
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
  }
};
