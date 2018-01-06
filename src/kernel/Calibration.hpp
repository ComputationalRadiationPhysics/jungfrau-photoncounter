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

        while (counter < numframes) {
            for (std::size_t i = 0; i < 3; ++i) {
                if (pede[(i * MAPSIZE) + id].counter == FRAMESPERSTAGE)
                    stage++;
            }

            while (counter < ((stage + 1u) * FRAMESPERSTAGE) &&
                   counter < ((stage * FRAMESPERSTAGE) + numframes)) {
                pede[(stage * MAPSIZE) + id].movAvg +=
                    data[(MAPSIZE * (counter)) + id +
                         (FRAMEOFFSET * (counter + 1u))] &
                    0x3fff;
                pede[(stage * MAPSIZE) + id].counter++;
                counter++;
            }
            if (pede[(stage * MAPSIZE) + id].counter == FRAMESPERSTAGE) {
                pede[(stage * MAPSIZE) + id].movAvg /= FRAMESPERSTAGE;
                pede[(stage * MAPSIZE) + id].value =
                    pede[(stage * MAPSIZE) + id].movAvg;
            }
        }
    }
};
