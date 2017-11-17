#include "../Config.hpp"


struct SummationKernel {
    template <typename TAcc,
              typename TData,
              typename TAmount,
              typename TNum,
              typename TSum>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc,
                                  TData const* const data,
                                  TAmount const amount,
                                  TNum const num,
                                  TSum* const sum) const -> void
    {
        auto const globalThreadIdx =
            alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        auto const globalThreadExtent =
            alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

        auto const linearizedGlobalThreadIdx =
            alpaka::idx::mapIdx<1u>(globalThreadIdx, globalThreadExtent);

        auto id = linearizedGlobalThreadIdx[0u];

        TSum summation = 0;

        for (std::size_t i = 0; i < num; ++i) {
            summation += data[(i * MAPSIZE) + id + ((i + 1u) * FRAMEOFFSET)];
            if (i % amount) {
                sum[((i / amount) * MAPSIZE) + id] = summation;
                summation = 0;
            }
        }
    }
};
