#pragma once
#include "helpers.hpp"

template<typename Config>
struct GainmapInversionKernel {
    template <typename TAcc, typename TGain>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc,
                                  TGain* const gainmaps) 
        const -> void
    {
        auto id = getLinearIdx(acc);
        
        // check range
        if (id >= Config::MAPSIZE)
            return;
        
        for (size_t i = 0; i < Config::GAINMAPS; ++i) {
            gainmaps[i][id] = 1.0 / gainmaps[i][id];
        }
    }
};

