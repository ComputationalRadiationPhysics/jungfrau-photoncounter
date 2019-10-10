#pragma once
#include "helpers.hpp"

template <typename Config> struct DriftMapKernel {
  template <typename TAcc, typename TInitPedestalMap, typename TPedestalMap,
            typename TDriftMap>
  ALPAKA_FN_ACC auto
  operator()(TAcc const &acc, TInitPedestalMap const *const initialPedestalMaps,
             TPedestalMap const *const pedestalMaps, TDriftMap *driftMaps) const
      -> void {
    auto globalId = getLinearIdx(acc);
    auto elementsPerThread = getLinearElementExtent(acc);

    // iterate over all elements in the thread
    for (auto id = globalId * elementsPerThread;
         id < (globalId + 1) * elementsPerThread; ++id) {

      // check range
      if (id >= Config::MAPSIZE)
        break;

      driftMaps->data[id] =
          pedestalMaps[0][id] - initialPedestalMaps[0][id].mean;
    }
  }
};
