#include "../Config.hpp"


struct SummationKernel {
    template <typename TAcc,
              typename TData,
              typename TNumFrames,
              typename TNumSumFrames,
              typename TSumMap>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc,
                                  TData const* const data,
                                  TNumFrames const numFrames,
                                  TNumSumFrames const numSumFrames,
                                  TSumMap* const sum) const -> void
    {
        auto const globalThreadIdx =
            alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        auto const globalThreadExtent =
            alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

        auto const linearizedGlobalThreadIdx =
            alpaka::idx::mapIdx<1u>(globalThreadIdx, globalThreadExtent);

        auto id = linearizedGlobalThreadIdx[0u];

        for (TNumFrames i = 0; i < numFrames; ++i) {
            sum[i / numSumFrames].data[id] += data[i].data[id];
        }
    }
};
