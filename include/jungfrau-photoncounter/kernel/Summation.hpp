#pragma once
#include "AlpakaHelper.hpp"

template <typename Config> struct SummationKernel {
  template <typename TAcc, typename TData, typename TNumFrames,
            typename TNumSumFrames, typename TSumMap>
  ALPAKA_FN_ACC auto operator()(TAcc const &acc, TData const *const data,
                                TNumSumFrames const numSumFrames,
                                TNumFrames const numFrames,
                                TSumMap *const sum) const -> void {
    auto globalId = getLinearIdx(acc);
    auto elementsPerThread = getLinearElementExtent(acc);

    // iterate over all elements in the thread
    for (auto id = globalId * elementsPerThread;
         id < (globalId + 1) * elementsPerThread; ++id) {

      // check range
      if (id >= Config::MAPSIZE)
        break;

      for (TNumFrames i = 0; i < numFrames; ++i) {
        if (i % numSumFrames == 0)
          sum[i / numSumFrames].data[id] = data[i].data[id];
        else
          sum[i / numSumFrames].data[id] += data[i].data[id];
      }
    }
  }
};
