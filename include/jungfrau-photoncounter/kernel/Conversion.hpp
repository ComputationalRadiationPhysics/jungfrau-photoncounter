#pragma once
#include "ForEach.hpp"
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

    // iterate over all frames
    for (TNumFrames i = 0; i < numFrames; ++i) {
      //! @todo: make this more dynamic later
      uint32_t workerSize = getLinearElementExtent(acc);

      // first thread copies frame header to output
      if (getLinearIdx(acc) == 0) {
        copyFrameHeader(detectorData[i], energyMaps[i]);
        copyFrameHeader(detectorData[i], gainStageMaps[i]);
      }

      /*
      uint32_t workerIdx = getLinearIdx(acc);
      uint32_t domainSize = Config::MAPSIZE;

      // iterate over whole extent
      uint32_t const iterationExtent =
          std::min(workerSize, domainSize - workerIdx * workerSize);

//#pragma omp simd
      for (uint32_t i = 0u; i < iterationExtent; ++i) {
        uint32_t const id = workerIdx * workerSize + i;

        // convert input data
        processInput(acc, detectorData[i], gainMaps, pedestalMaps,
                     initPedestalMaps, gainStageMaps[i], energyMaps[i], mask,
                     id, pedestalFallback);

        // read data from generated maps
        auto dataword = detectorData[i].data[id];
        auto adc = getAdc(dataword);
        const auto &gainStage = gainStageMaps[i].data[id];
        const auto &pedestal = pedestalMaps[gainStage][id];
        const auto &stddev = initPedestalMaps[gainStage][id].stddev;

        //! @todo: for all pixels in gain stage > 0 the dark pixel
        //! condition should never be satisfied

        // check "dark pixel" condition
        // if (pedestal - c * stddev <= adc && pedestal + c * stddev >= adc &&
        if (pedestal - stddev <= adc && pedestal + stddev >= adc &&
            !pedestalFallback) {
          updatePedestal(adc, Config::MOVING_STAT_WINDOW_SIZE,
                         pedestalMaps[gainStage][id]);
        }
      }*/

      // execute double loop to take advantage of SIMD
      forEach(getLinearIdx(acc), workerSize, Config::MAPSIZE, [&](uint32_t id) {
        // convert input data
        processInput(acc, detectorData[i], gainMaps, pedestalMaps,
                     initPedestalMaps, gainStageMaps[i], energyMaps[i], mask,
                     id, pedestalFallback);

        // read data from generated maps
        auto dataword = detectorData[i].data[id];
        auto adc = getAdc(dataword);
        const auto &gainStage = gainStageMaps[i].data[id];
        const auto &pedestal = pedestalMaps[gainStage][id];
        const auto &stddev = initPedestalMaps[gainStage][id].stddev;

        //! @todo: for all pixels in gain stage > 0 the dark pixel
        //! condition should never be satisfied

        // check "dark pixel" condition
        // if (pedestal - c * stddev <= adc && pedestal + c * stddev >= adc &&
        if (pedestal - stddev <= adc && pedestal + stddev >= adc &&
            !pedestalFallback && gainStage == 0) {
          updatePedestal(adc, Config::MOVING_STAT_WINDOW_SIZE,
                         pedestalMaps[gainStage][id]);
        }
      });

      /*uint32_t workerIdx = getLinearIdx(acc);
      uint32_t domainSize = Config::MAPSIZE;
        // iterate over whole extent
        uint32_t const iterationExtent =
            std::min(workerSize, domainSize - workerIdx * workerSize);

      #pragma omp simd
        for (uint32_t i = 0u; i < iterationExtent; ++i) {
          uint32_t const id = workerIdx * workerSize + i;

          // call functor// convert input data
          processInput(acc, detectorData[i], gainMaps, pedestalMaps,
                       initPedestalMaps, gainStageMaps[i], energyMaps[i], mask,
                       id, pedestalFallback);

          // read data from generated maps
          auto dataword = detectorData[i].data[id];
          auto adc = getAdc(dataword);
          const auto &gainStage = gainStageMaps[i].data[id];
          const auto &pedestal = pedestalMaps[gainStage][id];
          const auto &stddev = initPedestalMaps[gainStage][id].stddev;

          //! @todo: for all pixels in gain stage > 0 the dark pixel
          //! condition should never be satisfied

          // check "dark pixel" condition
          if (pedestal - c_double * stddev <= adc && pedestal + c_double *
      stddev >= adc && !pedestalFallback) { updatePedestal(adc,
      Config::MOVING_STAT_WINDOW_SIZE, pedestalMaps[gainStage][id]);
          }
        }*/
    }
  }
};
