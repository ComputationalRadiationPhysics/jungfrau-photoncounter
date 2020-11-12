#pragma once
#include "ForEach.hpp"
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

    auto calibrationLambda = [&](const uint64_t id) {
      const std::size_t FRAMESPERSTAGE[] = {Config::FRAMESPERSTAGE_G0,
                                            Config::FRAMESPERSTAGE_G1,
                                            Config::FRAMESPERSTAGE_G2};

      // find expected gain stage
      char expectedGainStage = 0;
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
    };

    // doesn't work with SIMD because of dependencies between frames
    // execute double loop to take advantage of SIMD
    // forEach(getLinearIdx(acc), getLinearElementExtent(acc), Config::MAPSIZE,
    //        calibrationLambda);

    // iterate over whole extent
    uint32_t const iterationExtent = std::min(
        getLinearElementExtent(acc),
        Config::MAPSIZE - getLinearIdx(acc) * getLinearElementExtent(acc));

    for (uint32_t i = 0u; i < iterationExtent; ++i) {
      uint32_t const localIdx =
          getLinearIdx(acc) * getLinearElementExtent(acc) + i;

      // call functor
      calibrationLambda(localIdx);
    }
  }
};
