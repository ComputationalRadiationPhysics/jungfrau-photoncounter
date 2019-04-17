#pragma once
#include "helpers.hpp"

template<typename Config>
struct CheckStdDevKernel {
    template <typename TAcc,
              typename TInitPedestalMap,
              typename TMaskMap,
              typename TRmsThreshold>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc,
                                  TInitPedestalMap const* const initPedestalMap,
                                  TMaskMap* const mask,
                                  TRmsThreshold const threshold) const -> void
    {
        auto id = getLinearIdx(acc);
        
        // check range
        if (id >= Config::MAPSIZE)
            return;

        // check if measured RMS exceeds threshold
        if (initPedestalMap[0][id].stddev >
            threshold * threshold * Config::MOVING_STAT_WINDOW_SIZE)
            mask->data[id] = false;
    }
};
