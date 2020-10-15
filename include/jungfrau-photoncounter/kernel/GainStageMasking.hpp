#pragma once
#include "ForEach.hpp"
#include "helpers.hpp"

template <typename Config> struct GainStageMaskingKernel {
  template <typename TAcc, typename TGainStageMap, typename TNumFrames,
            typename TMask>
  ALPAKA_FN_ACC auto
  operator()(TAcc const &acc, TGainStageMap *const inputGainStageMaps,
             TGainStageMap *outputGainStageMaps, TNumFrames const numFrames,
             TMask const *const mask) const -> void {

    // iterate over all frames
    for (TNumFrames i = 0; i < numFrames; ++i) {
      // copy frame header
      if (getLinearIdx(acc) == 0) {
        copyFrameHeader(inputGainStageMaps[i], outputGainStageMaps[i]);
      }

      auto maskingLambda = [&](const uint64_t id) {
        // use masks to check whether the channel is valid or masked out
        bool isValid = !mask ? 1 : mask->data[id];

        outputGainStageMaps[i].data[id] =
            (isValid ? inputGainStageMaps[i].data[id] : Config::MASKED_VALUE);
      };

      forEach(getLinearIdx(acc), getLinearElementExtent(acc), Config::MAPSIZE,
              maskingLambda);
    }
  }
};
