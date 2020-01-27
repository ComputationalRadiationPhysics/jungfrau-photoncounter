#pragma once
#include "helpers.hpp"

template <typename Config> struct ConversionKernel {
  template <typename TAcc, typename TDetectorData, typename TGainMap,
            typename TInitPedestalMap, typename TPedestalMap,
            typename TGainStageMap, typename TEnergyMap, typename TNumFrames,
            typename TMask, typename TNumStdDevs = int>
  ALPAKA_FN_ACC auto
  operator()(TAcc const &acc, TDetectorData const *const detectorData,
             TGainMap const *const gainMaps,
             TInitPedestalMap *const initPedestalMaps,
             TPedestalMap *const pedestalMaps,
             TGainStageMap *const gainStageMaps, TEnergyMap *const energyMaps,
             TNumFrames const numFrames, TMask const *const mask,
             bool pedestalFallback, TNumStdDevs const c = Config::C) const
      -> void {    
    auto globalId = getLinearIdx(acc);
    auto elementsPerThread = getLinearElementExtent(acc);

    // iterate over all elements in the thread
    for (auto id = globalId * elementsPerThread;
         id < (globalId + 1) * elementsPerThread; ++id) {
      // check range
      if (id >= Config::MAPSIZE)
        break;

      for (TNumFrames i = 0; i < numFrames; ++i) {
        processInput(acc, detectorData[i], gainMaps, pedestalMaps,
                     initPedestalMaps, gainStageMaps[i], energyMaps[i], mask,
                     id, pedestalFallback);

        // read data from generated maps
        auto dataword = detectorData[i].data[id];
        auto adc = getAdc(dataword);
        const auto &gainStage = gainStageMaps[i].data[id];
        const auto &pedestal = pedestalMaps[gainStage][id];
        const auto &stddev = initPedestalMaps[gainStage][id].stddev;

        // check "dark pixel" condition
        if (pedestal - c * stddev <= adc && pedestal + c * stddev >= adc &&
            !pedestalFallback) {
          updatePedestal(adc, Config::MOVING_STAT_WINDOW_SIZE,
                         pedestalMaps[gainStage][id]);
        }
      }
    }
  }
};
