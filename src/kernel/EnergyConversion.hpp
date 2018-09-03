#include "../Config.hpp"

struct EnergyConversionKernel {
    template <typename TAcc,
              typename TData,
              typename TGainMap,
              typename TStatistics,
              typename TNumFrames,
              typename TEnergyMap,
              typename TMask,
              >
    ALPAKA_FN_ACC auto operator()(TAcc const& acc,
                                  TData const* const data,
                                  TGain const* const gainMaps,
                                  TStatistics const* const statMaps,
                                  TNumFrames const numFrames,
                                  TEnergyMap* const energyMaps,
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

        uint16_t pedestal[3];
        uint16_t gain[3];

        for (std::size_t i = 0; i < 3; ++i) {
            pedestal[i] = statMaps[i][id].mean;
            gain[i] = gainMaps[i][id];
        }

        uint16_t dataword;
        uint16_t adc;
        float energy;
        bool m1;
        bool m2;

        for (std::size_t i = 0; i < num; ++i) {
            // first thread copies frame header
            if (id == 0) {
                energyMaps[i].frameNumber = data[i].frameNumber;
                energyMaps[i].bunchId = data[i].bunchId;
            }
            dataword = data[i].imagedata[id];
            m1 = mask[i][id];
            m2 = manualMask[id];

            // calculate energy only for pixels in mask, 0 otherwise
            if (m1 && m2) {
                adc = dataword & 0x3fff;

                switch ((dataword & 0xc000) >> 14) {
                case 0:
                    energy = (adc - pedestal[0]) / gain[0];
                    if (energy < 0)
                        energy = 0;
                    break;
                case 1:
                    energy = (-1) * (pedestal[1] - adc) / gain[1];
                    if (energy < 0)
                        energy = 0;
                    break;
                case 3:
                    energy = (-1) * (pedestal[2] - adc) / gain[2];
                    if (energy < 0)
                        energy = 0;
                    break;
                default:
                    energy = 0;
                    break;
                }
            }
            else {
                energy = 0;
            }
            // set final value
            energyMaps[i].data[id] = energy;
        }
    }
};
