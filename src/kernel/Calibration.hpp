#include "../Config.hpp"


struct CalibrationKernel {
    template <typename TAcc, typename TData, typename TPedestal, typename TNum>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc,
                                  TData const* const data,
                                  TPedestal* const pede,
                                  TNum const numframes) const -> void
    {
        auto const globalThreadIdx =
            alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        auto const globalThreadExtent =
            alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

        auto const linearizedGlobalThreadIdx =
            alpaka::idx::mapIdx<1u>(globalThreadIdx, globalThreadExtent);

        auto id = linearizedGlobalThreadIdx[0u];

        std::size_t counter = 0;
        uint16_t stage = 0;

        for (std::size_t i = 0; i < 3; ++i) {
            if (pede[i][id].counter == FRAMESPERSTAGE)
                stage = i;
        }

        while (counter < numframes) {
           stage++; 

            while (counter < ((stage + 1u) * FRAMESPERSTAGE) &&
                   counter < ((stage * FRAMESPERSTAGE) + numframes)) {
                pede[stage][id].movAvg +=
                    data[counter].imagedata[id] &0x3fff;
                pede[stage][id].counter++;
                counter++;
            }
            if (pede[stage][id].counter == FRAMESPERSTAGE) {
                pede[stage][id].movAvg /= FRAMESPERSTAGE;
                pede[stage][id].value =
                    pede[stage][id].movAvg;
            }
        }
    }
};
