#include "../Config.hpp"

struct EnergyConversionKernel {
    template <typename TAcc,
              typename TDetectorData,
              typename TGainMap,
              typename TStatistics,
              typename TNumFrames,
              typename TGainStageMap,
              typename TEnergyMap,
              typename TMask,
              >
    ALPAKA_FN_ACC auto operator()(TAcc const& acc,
                                  TDetectorData const* const detectorData,
                                  TGainMap const* const gainMaps,
                                  TStatistics const* const statMaps,
                                  TNumFrames const numFrames,
                                  TEnergyMap* const energyMaps,
                                  TGainStageMap* const gainStageMaps,
                                  TMask* const manualMask,
                                  TMask* const mask
                                  ) const -> void
    {
        auto const globalThreadIdx =
            alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        auto const globalThreadExtent =
            alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

        auto const linearizedGlobalThreadIdx =
            alpaka::idx::mapIdx<1u>(globalThreadIdx, globalThreadExtent);

        auto id = linearizedGlobalThreadIdx[0u];
        bool isValid = manualMask[id] && mask[id];
        // for each frame to be processed
        for (std::size_t i = 0; i < num; ++i) {
            auto dataword = detectorData[i].data[id];
            // first thread copies frame header
            if (id == 0) {
                auto header = detectorData[i].header;
                energyMaps[i].header = header;
                gainStageMaps[i].header = header;
            }
            std::uint16_t adc = dataword & 0x3fff;
            char gainStage = (dataword & 0xc000) >> 14;
            // map gain stages from 0, 1, 3 to 0, 1, 2
            if (gainStage == 3) {
                gainStage = 2;
            }
            gainStageMaps[i].data[id] = gainStage;
            energyMaps[i].data[id] = 
                (adc - statMaps[gainStage * MAPSIZE + id].mean) 
                    / gainMaps[gainStage][id];
            // set energy to zero if pixel is marked false by mask
            if (!isValid) {
                energyMaps[i].data[id] = 0;
            }
        }
    }
};
