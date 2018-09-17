#pragma once
#include "../Config.hpp"
#include "helpers.hpp"

struct ConversionKernel {
    template <typename TAcc,
              typename TDetectorData,
              typename TGainMap,
              typename TPedestalMap,
              typename TGainStageMap,
              typename TEnergyMap,
              typename TNumFrames,
              typename TMask>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc,
                                  TDetectorData const* const detectorData,
                                  TGainMap const* const gainMaps,
                                  TPedestalMap* const pedestalMaps,
                                  TGainStageMap* const gainStageMaps,
                                  TEnergyMap* const energyMaps,
                                  TNumFrames const numFrames,
                                  TMask const* const mask) const -> void
    {
        auto const globalThreadIdx =
            alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        auto const globalThreadExtent =
            alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

        auto const linearizedGlobalThreadIdx =
            alpaka::idx::mapIdx<1u>(globalThreadIdx, globalThreadExtent);

        auto id = linearizedGlobalThreadIdx[0u];

        // use masks to check whether the channel is valid or masked out
        bool isValid = mask->data[id];

        for (TNumFrames i = 0; i < numFrames; ++i) {
          if(id == 1024 * 512 - 1) {
            printf("0");
          }
            auto dataword = detectorData[i].data[id];
            auto adc = getAdc(dataword);
          
          if(id == 1024 * 512 - 1) {
            printf("1");
          }
            auto& gainStage = gainStageMaps[i].data[id];
            gainStage = getGainStage(dataword);


          if(id == 1024 * 512 - 1) {
            printf("2");
          }


  
            /*
            // first thread copies frame header to output maps
            if (id == 0) {
                copyFrameHeader(detectorData[i], energyMaps[i]);
                copyFrameHeader(detectorData[i], gainStageMaps[i]);
            }
            */
            
            const auto& pedestal = pedestalMaps[gainStage][id].mean;
            const auto& gain = gainMaps[gainStage][id];

          if(id == 1024 * 512 - 1) {
            printf("3");
          }
            
            // calculate energy of current channel
            auto& energy = energyMaps[i].data[id];
            energy = (adc - pedestal) / gain;
            
            
            if(id == 1024 * 512 - 1) {
              printf("%d, %d, %d, %f, %f\n", dataword, adc, gainStage, pedestal, gain);
            }
            
            /*
            // calculate energy of current channel
            auto& energy = energyMaps[i].data[id];
            energy = (adc - pedestal) / gain;

            // set energy to zero if masked out
            if (!isValid)
                energy = 0;
            */
        }
            
    }
};
