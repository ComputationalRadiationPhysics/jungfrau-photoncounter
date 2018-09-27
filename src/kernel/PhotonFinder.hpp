#pragma once
#include "../Config.hpp"
#include "helpers.hpp"

struct PhotonFinderKernel {
    template <typename TAcc,
              typename TDetectorData,
              typename TGainMap,
              typename TPedestalMap,
              typename TGainStageMap,
              typename TEnergyMap,
              typename TPhotonMap,
              typename TNumFrames,
              typename TMask,
              typename TNumStdDevs = int>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc,
                                  TDetectorData const* const detectorData,
                                  TGainMap const* const gainMaps,
                                  TPedestalMap* const pedestalMaps,
                                  TGainStageMap* const gainStageMaps,
                                  TEnergyMap* const energyMaps,
                                  TPhotonMap* const photonMaps,
                                  TNumFrames const numFrames,
                                  TMask* const mask,
                                  TNumStdDevs const c = 5) const -> void
    {
        auto const globalThreadIdx =
            alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        auto const globalThreadExtent =
            alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

        auto const linearizedGlobalThreadIdx =
            alpaka::idx::mapIdx<1u>(globalThreadIdx, globalThreadExtent);

        auto id = linearizedGlobalThreadIdx[0u];

        for (TNumFrames i = 0; i < numFrames; ++i) {
            // generate energy maps and gain stage maps
            processInput(acc, 
                    detectorData[i], 
                    gainMaps, 
                    pedestalMaps, 
                    gainStageMaps[i],
                    energyMaps[i],
                    mask,
                    id);
            // read data from generated maps
            auto dataword = detectorData[i].data[id];
            auto adc = getAdc(dataword);

            const auto& gainStage = gainStageMaps[i].data[id];
            //printf("%d ", gainStage);
            // first thread copies frame header to output
            if (id == 0) {
                copyFrameHeader(detectorData[i], photonMaps[i]);
            }

            const auto& energy = energyMaps[i].data[id];
            auto& photonCount = photonMaps[i].data[id];

            // calculate photon count from calibrated energy
            photonCount = (energy + BEAMCONST) * PHOTONCONST;

            const auto& pedestal = pedestalMaps[gainStage][id].mean;
            const auto& stddev = pedestalMaps[gainStage][id].stddev;

            // check "dark pixel" condition
            if (pedestal - c * stddev <= adc && pedestal + c * stddev >= adc) {
              updatePedestal(acc, adc, pedestalMaps[gainStage][id]);
            }
        }
    }
};
