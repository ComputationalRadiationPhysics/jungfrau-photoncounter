#pragma once
#include "../Config.hpp"
#include "helpers.hpp"

struct DriftMapKernel {
    template <typename TAcc,
              typename TInitPedestalMap,
              typename TPedestalMap,
              typename TDriftMap>
    ALPAKA_FN_ACC auto
    operator()(TAcc const& acc,
               TInitPedestalMap const* const initialPedestalMaps,
               TPedestalMap const* const pedestalMaps,
               TDriftMap* driftMaps) const -> void
    {
        auto id = getLinearIdx(acc);
        
        // check range
        if (id >= MAPSIZE)
            return;

        driftMaps->data[id] =
            pedestalMaps[0][id] - initialPedestalMaps[0][id].mean;
    }
};
