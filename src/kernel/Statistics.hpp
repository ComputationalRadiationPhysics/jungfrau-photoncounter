#include "../Config.hpp"
#include <cmath>


struct StatisticsKernel {
    template <typename TAcc, 
              typename TData, 
              typename TGain, 
              typename TNum,
              typename TPedestal,
              typename TMask>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, 
                                  TData* const data,
                                  TGain const* const gainmap,
                                  TNum const num,
                                  TPedestal* const pedestal,
                                  TMask* const mask) const -> void
    {
        auto const globalThreadIdx =
            alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        auto const globalThreadExtent =
            alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

        auto const linearizedGlobalThreadIdx =
            alpaka::idx::mapIdx<1u>(globalThreadIdx, globalThreadExtent);

        auto id = linearizedGlobalThreadIdx[0u];

        //charge correction
        TGain gain[3];
        uint16_t dataword;
        uint16_t adc;
        double charge;

        for (std::size_t i = 0; i < 3; i++) {
            gain[i] = gainmap[(i * MAPSIZE) + id];
        }
       
        double delta;
        double delta2;
        std::size_t counter;

        for (std::size_t i = 0; i < num; ++i) { 
            dataword = data[(MAPSIZE * i) + id + (FRAMEOFFSET * (i + 1u))];
            adc = dataword & 0x3fff;
            uint8_t stage = ((dataword & 0xbfff) >> 14);
            if(stage == 3) stage = 2;
            charge = adc / gain[stage];
            
            if(pedestal[(stage * MAPSIZE) + id].counter < 1000) {
                pedestal[(stage * MAPSIZE) + id].counter += 1;
                counter = pedestal[(stage * MAPSIZE) + id].counter;
                
                //mean algorithm by Welford
                delta = charge - pedestal[(stage * MAPSIZE) +id].mean;
                pedestal[(stage * MAPSIZE) +id].mean += (delta/counter);
                delta2 = charge - pedestal[(stage * MAPSIZE) +id].mean;

                pedestal[(stage * MAPSIZE) +id].M2 += delta * delta2;

                pedestal[(stage * MAPSIZE) +id].stddev = 
                    sqrt(pedestal[(stage * MAPSIZE) +id].M2 / counter);
            } else if (dataword < (pedestal[(stage * MAPSIZE) +id].mean +
                (pedestal[(stage * MAPSIZE) +id].stddev * 5)) ) { 
                pedestal[(stage * MAPSIZE) + id].counter += 1;
                counter = pedestal[(stage * MAPSIZE) + id].counter;
                
                //mean algorithm by Welford
                delta = charge - pedestal[(stage * MAPSIZE) +id].mean;
                pedestal[(stage * MAPSIZE) +id].mean += (delta/counter);
                delta2 = charge - pedestal[(stage * MAPSIZE) +id].mean;

                pedestal[(stage * MAPSIZE) +id].M2 += delta * delta2;

                pedestal[(stage * MAPSIZE) +id].stddev = 
                    sqrt(pedestal[(stage * MAPSIZE) +id].M2 / counter);
                data[(MAPSIZE * i) + id + (FRAMEOFFSET * (i + 1u))] = -1;

                //set masking pixel
                mask[(MAPSIZE * i) + id] = false;
            } else {
                //set masking pixel
                mask[(MAPSIZE * i) + id] = true;
            }
        }
    }
};