#pragma once
#include "ForEach.hpp"
#include "helpers.hpp"

template <typename Config> struct GainmapInversionKernel {
  template <typename TAcc, typename TGain>
  ALPAKA_FN_ACC auto operator()(TAcc const &acc, TGain *const gainmaps) const
      -> void {
    auto inversionLambda = [=] ALPAKA_FN_ACC(const uint64_t id) {
      for (size_t i = 0; i < Config::GAINMAPS; ++i) {
        gainmaps[i][id] = 1.0 / gainmaps[i][id];
      }
    };

    // iterate over all elements in the thread
    forEach(getLinearIdx(acc), getLinearElementExtent(acc), Config::MAPSIZE,
            inversionLambda);
  }
};
