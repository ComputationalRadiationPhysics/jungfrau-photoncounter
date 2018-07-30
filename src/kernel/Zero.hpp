#include "../Config.hpp"

struct ZeroKernel {
    template <typename TAcc,
              typename TPedestal>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc,
                                  TPedestal* const pedestal) const -> void
    {
        auto const globalThreadIdx =
            alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        auto const globalThreadExtent =
            alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

        auto const linearizedGlobalThreadIdx =
            alpaka::idx::mapIdx<1u>(globalThreadIdx, globalThreadExtent);

        auto id = linearizedGlobalThreadIdx[0u];

        for(std::size_t i = 0; i < 3; ++i) {
            pedestal[i][id].counter = 0;
            pedestal[i][id].mean = 0;
            pedestal[i][id].M2 = 0;
            pedestal[i][id].stddev = 0;
        } 
    }
};
