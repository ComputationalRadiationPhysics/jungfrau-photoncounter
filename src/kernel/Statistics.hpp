#include "../Config.hpp"
#include <cmath>


struct StatisticsKernel {
    template <typename TAcc, 
              typename TData, 
              typename TNum,
              typename TPedestal,
              typename TMask>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, 
                                  TData* const data,
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
        uint16_t dataword;
        uint16_t charge;

        double delta;
        double delta2;
        std::size_t counter;

        for (std::size_t i = 0; i < num; ++i) { 
            dataword = data[i].imagedata[id];
            charge = dataword & 0x3fff;
            uint8_t stage = (dataword >> 14);
            if(stage == 3) stage = 2;
            
            if(pedestal[stage][id].counter < 1000) {
                pedestal[stage][id].counter += 1;
                counter = pedestal[stage][id].counter;
                
                //mean algorithm by Welford
                delta = charge - pedestal[stage][id].mean;
                pedestal[stage][id].mean += (delta/counter);
                delta2 = charge - pedestal[stage][id].mean;

                pedestal[stage][id].M2 += delta * delta2;

                pedestal[stage][id].stddev = 
                    sqrt(pedestal[stage][id].M2 / counter);
            } else if (dataword < (pedestal[stage][id].mean +
                (pedestal[stage][id].stddev * 5)) ) { 
                pedestal[stage][id].counter += 1;
                counter = pedestal[stage][id].counter;
                
                //mean algorithm by Welford
                delta = charge - pedestal[stage][id].mean;
                pedestal[stage][id].mean += (delta/counter);
                delta2 = charge - pedestal[stage][id].mean;

                pedestal[stage][id].M2 += delta * delta2;

                pedestal[stage][id].stddev = 
                    sqrt(pedestal[stage][id].M2 / counter);

                //set masking pixel
                mask[i][id] = false;
            } else {
                //set masking pixel
                mask[i][id] = true;
            }
        }
    }
};
