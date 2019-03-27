#pragma once

#include <alpaka/alpaka.hpp>

struct MaxValueCopyKernel {
    ALPAKA_NO_HOST_ACC_WARNING

    template <typename TAcc, typename TEnergyMap, typename TEnergyValue, typename TNumFrames>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc,
                                  TEnergyMap const* const source,
                                  TEnergyValue* destination,
                                  TNumFrames const& numFrames) const -> void
    {
        auto id = getLinearIdx(acc);
        
        // check range
        if (id >= MAPSIZE)
            return;
        
        if(id < numFrames) {
          destination[id] = source[id].data[0];
        }
    }
};
