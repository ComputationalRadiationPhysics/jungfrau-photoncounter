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
        auto const globalThreadIdx =
            alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        auto const globalThreadExtent =
            alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

        auto const linearizedGlobalThreadIdx =
            alpaka::idx::mapIdx<1u>(globalThreadIdx, globalThreadExtent);

        auto id = linearizedGlobalThreadIdx[0u];

        // check range
        if (id >= MAPSIZE)
            return;
        
        if(id < numFrames) {
          destination[id] = source[id].data[0];
        }
    }
};
