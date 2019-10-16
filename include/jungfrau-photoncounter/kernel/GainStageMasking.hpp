#pragma once
#include "helpers.hpp"

template <typename Config> struct GainStageMaskingKernel {
  template <typename TAcc, typename TGainStageMap, typename TNumFrames,
            typename TMask>
  ALPAKA_FN_ACC auto
  operator()(TAcc const &acc, TGainStageMap *const inputGainStageMaps,
             TGainStageMap *outputGainStageMaps, TNumFrames const numFrames,
             TMask const *const mask) const -> void {
    auto globalId = getLinearIdx(acc);
    auto elementsPerThread = getLinearElementExtent(acc);

    // iterate over all elements in the thread
    for (auto id = globalId * elementsPerThread;
         id < (globalId + 1) * elementsPerThread; ++id) {

      // check range
      if (id >= Config::MAPSIZE)
        break;

      // use masks to check whether the channel is valid or masked out
      bool isValid = !mask ? 1 : mask->data[id];

      for (TNumFrames i = 0; i < numFrames; ++i) {
        outputGainStageMaps[i].data[id] =
            (isValid ? inputGainStageMaps[i].data[id] : Config::MASKED_VALUE);
      }
    }
  }
};
