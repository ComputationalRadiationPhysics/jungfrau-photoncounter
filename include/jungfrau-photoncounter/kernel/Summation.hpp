#pragma once
#include "AlpakaHelper.hpp"

template<typename Config>
struct SummationKernel {
    template <typename TAcc,
              typename TData,
              typename TNumFrames,
              typename TNumSumFrames,
              typename TSumMap>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc,
                                  TData const* const data,
                                  TNumSumFrames const numSumFrames,
                                  TNumFrames const numFrames,
                                  TSumMap* const sum) const -> void
    {
        auto id = getLinearIdx(acc);
        
        // check range
        if (id >= Config::MAPSIZE)
            return;
        
        for (TNumFrames i = 0; i < numFrames; ++i) {
            if (i % numSumFrames == 0)
                sum[i / numSumFrames].data[id] = data[i].data[id];
            else
                sum[i / numSumFrames].data[id] += data[i].data[id];
        }
    }
};
