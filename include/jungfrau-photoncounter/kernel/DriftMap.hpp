#pragma once
#include "../Config.hpp"
#include "helpers.hpp"

struct DriftMapKernel {
    template <typename TAcc,
              typename TPedestalMap,
              typename TDriftMap>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc,
                                  TPedestalMap const* const initialPedestalMaps,
                                  TPedestalMap const* const pedestalMaps,
                                  TDriftMap *driftMaps) const -> void
    {
        auto const globalThreadIdx =
            alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        auto const globalThreadExtent =
            alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

        auto const linearizedGlobalThreadIdx =
            alpaka::idx::mapIdx<1u>(globalThreadIdx, globalThreadExtent);

        auto id = linearizedGlobalThreadIdx[0u];
        
        driftMaps->data[id] = pedestalMaps[0][id].mean - initialPedestalMaps[0][id].mean;
    }
};
