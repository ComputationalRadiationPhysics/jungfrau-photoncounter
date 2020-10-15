#pragma once
#include "AlpakaHelper.hpp"
#include "ForEach.hpp"

template <typename Config> struct SummationKernel {
  template <typename TAcc, typename TData, typename TNumFrames,
            typename TNumSumFrames, typename TSumMap>
  ALPAKA_FN_ACC auto operator()(TAcc const &acc, TData const *const data,
                                TNumSumFrames const numSumFrames,
                                TNumFrames const numFrames,
                                TSumMap *const sum) const -> void {

    for (TNumFrames i = 0; i < numFrames; ++i) {
      auto sumLambda = [&](const uint64_t id) {
        if (i % numSumFrames == 0)
          sum[i / numSumFrames].data[id] = data[i].data[id];
        else
          sum[i / numSumFrames].data[id] += data[i].data[id];
      };

      // iterate over all elements in the thread
      forEach(getLinearIdx(acc), getLinearElementExtent(acc), Config::MAPSIZE,
              sumLambda);
    }
  }
};
