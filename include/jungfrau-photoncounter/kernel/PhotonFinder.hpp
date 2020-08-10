#pragma once
#include "helpers.hpp"

template <typename Config> struct PhotonFinderKernel {
  template <typename TAcc, typename TDetectorData, typename TGainMap,
            typename TInitPedestalMap, typename TPedestalMap,
            typename TGainStageMap, typename TEnergyMap, typename TPhotonMap,
            typename TNumFrames, typename TMask, typename TNumStdDevs = int>
  ALPAKA_FN_ACC auto operator()(
      TAcc const &acc, TDetectorData const *const detectorData,
      TGainMap const *const gainMaps, TInitPedestalMap *const initPedestalMaps,
      TPedestalMap *const pedestalMaps, TGainStageMap *const gainStageMaps,
      TEnergyMap *const energyMaps, TPhotonMap *const photonMaps,
      TNumFrames const numFrames, TMask *const mask, bool pedestalFallback,
      TNumStdDevs const c = Config::C) const -> void {
    auto id = getLinearIdx(acc);

    // check range
    if (id >= Config::MAPSIZE)
      return;

    for (TNumFrames i = 0; i < numFrames; ++i) {
      // generate energy maps and gain stage maps
      processInput(acc, detectorData[i], gainMaps, pedestalMaps,
                   initPedestalMaps, gainStageMaps[i], energyMaps[i], mask, id,
                   pedestalFallback);

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
      photonCount = (energy + Config::BEAMCONST) * Config::PHOTONCONST;

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
};
