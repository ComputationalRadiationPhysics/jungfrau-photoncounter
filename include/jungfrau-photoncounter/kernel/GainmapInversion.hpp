#pragma once
#include "helpers.hpp"

template <typename Config> struct GainmapInversionKernel {
  template <typename TAcc, typename TGain>
  ALPAKA_FN_ACC auto operator()(TAcc const &acc, TGain *const gainmaps) const
      -> void {
    auto globalId = getLinearIdx(acc);
    auto elementsPerThread = getLinearElementExtent(acc);
    
    // iterate over all elements in the thread
    for (auto id = globalId * elementsPerThread;
         id < (globalId + 1) * elementsPerThread; ++id) {
      // check range
      if (id >= Config::MAPSIZE)
        break;
      
     for (size_t i = 0; i < Config::GAINMAPS; ++i) {
        gainmaps[i][id] = 1.0 / gainmaps[i][id];
      }
    }
  }
};
