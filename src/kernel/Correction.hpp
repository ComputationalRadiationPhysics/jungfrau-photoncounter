#include "../Config.hpp"


struct CorrectionKernel {
    template <typename TAcc,
              typename TData,
              typename TPedestal,
              typename TGain,
              typename TNum,
              typename TPhoton,
              typename TMask>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc,
                                  TData const* const data,
                                  TPedestal* const pede,
                                  TGain const* const gainmap,
                                  TNum const num,
                                  TPhoton* const photon,
                                  TMask* const manualMask,
                                  TMask* const mask) const -> void
    {
        auto const globalThreadIdx =
            alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        auto const globalThreadExtent =
            alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

        auto const linearizedGlobalThreadIdx =
            alpaka::idx::mapIdx<1u>(globalThreadIdx, globalThreadExtent);

        auto id = linearizedGlobalThreadIdx[0u];


        TPhoton pedestal[3];
        TGain gain[3];

        for (std::size_t i = 0; i < 3; i++) {
            pedestal[i] = pede[(i * MAPSIZE) + id].mean;
            gain[i] = gainmap[(i * MAPSIZE) + id];
        }

        uint16_t dataword;
        uint16_t adc;
        float energy;
        bool m1;
        bool m2;

        for (std::size_t i = 0; i < num; ++i) {
            dataword = data[(MAPSIZE * i) + id + (FRAMEOFFSET * (i + 1u))];
            m1 = mask[(MAPSIZE * i) + id];
            m2 = manualMask[(MAPSIZE * i) + id];

            if(m1 && m2) {
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
                photon[(MAPSIZE * i) + id + (FRAMEOFFSET * (i + 1u))] =
                    int((energy + BEAMCONST) * PHOTONCONST);

                // copy the header
                if (globalThreadIdx[0u] < 8) {
                    photon[(MAPSIZE * i) + (globalThreadIdx[0u] * (i + 1u))] =
                        data[(MAPSIZE * i) + (globalThreadIdx[0u] * (i + 1u))];
                }
            }
        }
    }
};
