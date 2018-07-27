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

        for (decltype(id) i = 0; i < num; ++i) {
            sum[i / amount].imagedata[id] += data[i].imagedata[id];
        }
    }
};
