#pragma once
#include "helpers.hpp"

/*
template <typename Config> struct PhotonFinderKernel {
  template <typename TAcc, typename TDetectorData, typename TGainMap,
            typename TInitPedestalMap, typename TPedestalMap,
            typename TGainStageMap, typename TEnergyMap, typename TPhotonMap,
            typename TNumFrames, typename TBeamConst, typename TMask,
            typename TNumStdDevs = int>
  ALPAKA_FN_ACC auto operator()(
      TAcc const &acc, TDetectorData const *const detectorData,
      TGainMap const *const gainMaps, TInitPedestalMap *const initPedestalMaps,
      TPedestalMap *const pedestalMaps, TGainStageMap *const gainStageMaps,
      TEnergyMap *const energyMaps, TPhotonMap *const photonMaps,
      TNumFrames const numFrames, TBeamConst beamConst, TMask *const mask,
      bool pedestalFallback, TNumStdDevs const c = Config::C) const -> void {

    // iterate over all frames
    for (TNumFrames i = 0; i < numFrames; ++i) {
      uint32_t workerSize = getLinearElementExtent(acc);

      // calculate offset to achieve aligned access
      uint32_t const alignOffset = alignmentOffset(
          64, energyMaps[i].data,
          Config::MAPSIZE); // 24; //(64 - Config::FRAME_HEADER_SIZE) /
                            // sizeof(typename Config::DetectorData);

      auto photonLambda = [&](uint32_t id, uint16_t alignOffset) {
        // incorporate alignment offset
        id += alignOffset;

        // generate energy maps and gain stage maps
        processInput(acc, detectorData[i], gainMaps, pedestalMaps,
                     initPedestalMaps, gainStageMaps[i], energyMaps[i], mask,
                     id, pedestalFallback);
        const auto &gainStage = gainStageMaps[i].data[id];

        const auto &energy = energyMaps[i].data[id];
        auto &photonCount = photonMaps[i].data[id];

        // calculate photon count from calibrated energy
        double photonCountFloat = (static_cast<double>(energy) +
                                   static_cast<double>(beamConst) / 2.0) /
                                  static_cast<double>(beamConst);
        photonCount = (photonCountFloat < 0.0f) ? 0 : photonCountFloat;
      };

      auto photonLambda2 = [&](uint32_t id, uint16_t alignOffset) {
        // incorporate alignment offset
        id += alignOffset;

        // read data from generated maps
        auto dataword = detectorData[i].data[id];
        auto adc = getAdc(dataword);

        const auto &pedestal = pedestalMaps[gainStage][id];
        const auto &stddev = initPedestalMaps[gainStage][id].stddev;

        //! @todo: dark condition should only occur in gain stages > 0

        // check "dark pixel" condition
        if (pedestal - c * stddev <= adc && pedestal + c * stddev >= adc &&
            !pedestalFallback) {
          updatePedestal(adc, Config::MOVING_STAT_WINDOW_SIZE,
                         pedestalMaps[gainStage][id]);
        }
      };

      // first thread copies frame header to output
      if (getLinearIdx(acc) == 0) {
        copyFrameHeader(detectorData[i], photonMaps[i]);
        copyFrameHeader(detectorData[i], energyMaps[i]);
        copyFrameHeader(detectorData[i], gainStageMaps[i]);

        // handle alignment offset
        // for (uint32_t i = 0; i < alignOffset; ++i)
        //  photonLambda(i, 0);
      }

      // execute double loop to take advantage of SIMD
      forEach(getLinearIdx(acc), workerSize, Config::MAPSIZE, photonLambda, 0);
    }
  }
};*/

template <typename Config> struct PhotonFinderKernel {
  template <typename TAcc, typename TDetectorData, typename TGainMap,
            typename TInitPedestalMap, typename TPedestalMap,
            typename TGainStageMap, typename TEnergyMap, typename TPhotonMap,
            typename TNumFrames, typename TBeamConst, typename TMask,
            typename TNumStdDevs = int>
  ALPAKA_FN_ACC auto operator()(
      TAcc const &acc, TDetectorData const *const detectorData,
      TGainMap const *const gainMaps, TInitPedestalMap *const initPedestalMaps,
      TPedestalMap *const pedestalMaps, TGainStageMap *const gainStageMaps,
      TEnergyMap *const energyMaps, TPhotonMap *const photonMaps,
      TNumFrames const numFrames, TBeamConst beamConst, TMask *const mask,
      bool pedestalFallback, TNumStdDevs const c = Config::C) const -> void {

    // iterate over all frames
    for (TNumFrames i = 0; i < numFrames; ++i) {
      uint32_t workerSize = getLinearElementExtent(acc);

      auto photonLambda = [&](uint32_t id) {
        // generate energy maps and gain stage maps
        processInput(acc, detectorData[i], gainMaps, pedestalMaps,
                     initPedestalMaps, gainStageMaps[i], energyMaps[i], mask,
                     id, pedestalFallback);

        // read data from generated maps
        auto dataword = detectorData[i].data[id];
        auto adc = getAdc(dataword);
        const auto &gainStage = gainStageMaps[i].data[id];

        const auto &energy = energyMaps[i].data[id];
        auto &photonCount = photonMaps[i].data[id];

        // calculate photon count from calibrated energy
        double photonCountFloat = (static_cast<double>(energy) +
                                   static_cast<double>(beamConst) / 2.0) /
                                  static_cast<double>(beamConst);
        photonCount = (photonCountFloat < 0.0f) ? 0 : photonCountFloat;

        const auto &pedestal = pedestalMaps[gainStage][id];
        const auto &stddev = initPedestalMaps[gainStage][id].stddev;

        //! @todo: dark condition should only occur in gain stages > 0

        // check "dark pixel" condition
        if (pedestal - c * stddev <= adc && pedestal + c * stddev >= adc &&
            !pedestalFallback) {
          updatePedestal(adc, Config::MOVING_STAT_WINDOW_SIZE,
                         pedestalMaps[gainStage][id]);
        }
      };

      // first thread copies frame header to output
      if (getLinearIdx(acc) == 0) {
        copyFrameHeader(detectorData[i], photonMaps[i]);
        copyFrameHeader(detectorData[i], energyMaps[i]);
        copyFrameHeader(detectorData[i], gainStageMaps[i]);
      }

      // execute double loop to take advantage of SIMD
      forEach(getLinearIdx(acc), workerSize, Config::MAPSIZE, photonLambda);
    }
  }
};

/*
 *
template <typename Config> struct PhotonFinderKernel {
  template <typename TAcc, typename TDetectorData, typename TGainMap,
            typename TInitPedestalMap, typename TPedestalMap,
            typename TGainStageMap, typename TEnergyMap, typename TPhotonMap,
            typename TNumFrames, typename TBeamConst, typename TMask,
            typename TNumStdDevs = int>
  ALPAKA_FN_ACC auto operator()(
      TAcc const &acc, TDetectorData const *const detectorData,
      TGainMap const *const gainMaps, TInitPedestalMap *const initPedestalMaps,
      TPedestalMap *const pedestalMaps, TGainStageMap *const gainStageMaps,
      TEnergyMap *const energyMaps, TPhotonMap *const photonMaps,
      TNumFrames const numFrames, TBeamConst beamConst, TMask *const mask,
      bool pedestalFallback, TNumStdDevs const c = Config::C) const -> void {

    // iterate over all frames
    for (TNumFrames i = 0; i < numFrames; ++i) {
      uint32_t workerSize = getLinearElementExtent(acc);

      // calculate offset to achieve aligned access
      uint32_t const alignOffset = alignmentOffset(
          64, energyMaps[i].data,
          Config::MAPSIZE); // 24; //(64 - Config::FRAME_HEADER_SIZE) /
                            // sizeof(typename Config::DetectorData);

      auto photonLambda = [&](uint32_t id, uint32_t alignOffset) {
        // incorporate alignment offset
        id += alignOffset;

        // generate energy maps and gain stage maps
        processInput(acc, detectorData[i], gainMaps, pedestalMaps,
                     initPedestalMaps, gainStageMaps[i], energyMaps[i], mask,
                     id, pedestalFallback);

        // read data from generated maps
        auto dataword = detectorData[i].data[id];
        auto adc = getAdc(dataword);
        const auto &gainStage = gainStageMaps[i].data[id];

        const auto &energy = energyMaps[i].data[id];
        auto &photonCount = photonMaps[i].data[id];

        // calculate photon count from calibrated energy
        double photonCountFloat = (static_cast<double>(energy) +
                                   static_cast<double>(beamConst) / 2.0) /
                                  static_cast<double>(beamConst);
        photonCount = (photonCountFloat < 0.0f) ? 0 : photonCountFloat;

        const auto &pedestal = pedestalMaps[gainStage][id];
        const auto &stddev = initPedestalMaps[gainStage][id].stddev;

        //! @todo: dark condition should only occur in gain stages > 0

        // check "dark pixel" condition
        if (pedestal - c * stddev <= adc && pedestal + c * stddev >= adc &&
            !pedestalFallback) {
          updatePedestal(adc, Config::MOVING_STAT_WINDOW_SIZE,
                         pedestalMaps[gainStage][id]);
        }
      };

      // first thread copies frame header to output
      if (getLinearIdx(acc) == 0) {
        copyFrameHeader(detectorData[i], photonMaps[i]);
        copyFrameHeader(detectorData[i], energyMaps[i]);
        copyFrameHeader(detectorData[i], gainStageMaps[i]);

        // handle alignment offset
        for (uint32_t i = 0; i < alignOffset; ++i)
          photonLambda(i, 0);
      }

      // execute double loop to take advantage of SIMD
      forEach(getLinearIdx(acc), workerSize, Config::MAPSIZE - alignOffset,
              photonLambda, alignOffset);
    }
  }
};
 *
 *
 *
auto globalId = getLinearIdx(acc);
auto elementsPerThread = getLinearElementExtent(acc);

// iterate over all elements in the thread
for (auto id = globalId * elementsPerThread;
   id < (globalId + 1) * elementsPerThread; ++id) {

// check range
if (id >= Config::MAPSIZE)
  break;

for (TNumFrames i = 0; i < numFrames; ++i) {
  // generate energy maps and gain stage maps
  processInput(acc, detectorData[i], gainMaps, pedestalMaps,
               initPedestalMaps, gainStageMaps[i], energyMaps[i], mask,
               id, pedestalFallback);

  // read data from generated maps
  auto dataword = detectorData[i].data[id];
  auto adc = getAdc(dataword);
  const auto &gainStage = gainStageMaps[i].data[id];

  // first thread copies frame header to output
  if (id == 0) {
    copyFrameHeader(detectorData[i], photonMaps[i]);
  }

  const auto &energy = energyMaps[i].data[id];
  auto &photonCount = photonMaps[i].data[id];

  // calculate photon count from calibrated energy
  double photonCountFloat = (static_cast<double>(energy) +
                             static_cast<double>(beamConst) / 2.0) /
                            static_cast<double>(beamConst);
  photonCount = (photonCountFloat < 0.0f) ? 0 : photonCountFloat;

  const auto &pedestal = pedestalMaps[gainStage][id];
  const auto &stddev = initPedestalMaps[gainStage][id].stddev;

  //! @todo: dark condition should only occur in gain stages > 0

  // check "dark pixel" condition
  if (pedestal - c * stddev <= adc && pedestal + c * stddev >= adc &&
      !pedestalFallback) {
    updatePedestal(adc, Config::MOVING_STAT_WINDOW_SIZE,
                   pedestalMaps[gainStage][id]);
  }
}
}
}
};*/
