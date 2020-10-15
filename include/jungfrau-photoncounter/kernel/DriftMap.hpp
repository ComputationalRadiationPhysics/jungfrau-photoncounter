#pragma once
#include "ForEach.hpp"
#include "helpers.hpp"

template <typename Config> struct DriftMapKernel {
  template <typename TAcc, typename TInitPedestalMap, typename TPedestalMap,
            typename TDriftMap>
  ALPAKA_FN_ACC auto
  operator()(TAcc const &acc, TInitPedestalMap const *const initialPedestalMaps,
             TPedestalMap const *const pedestalMaps, TDriftMap *driftMaps) const
      -> void {
    auto driftLambda = [&](const uint64_t id) {
      driftMaps->data[id] =
          pedestalMaps[0][id] - initialPedestalMaps[0][id].mean;
    };

    // iterate over all elements in the thread
    forEach(getLinearIdx(acc), getLinearElementExtent(acc), Config::MAPSIZE,
            driftLambda);
  }
};
