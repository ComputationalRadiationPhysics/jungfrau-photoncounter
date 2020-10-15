#pragma once
#include "ForEach.hpp"
#include "helpers.hpp"

template <typename Config> struct CheckStdDevKernel {
  template <typename TAcc, typename TInitPedestalMap, typename TMaskMap,
            typename TRmsThreshold>
  ALPAKA_FN_ACC auto operator()(TAcc const &acc,
                                TInitPedestalMap const *const initPedestalMap,
                                TMaskMap *const mask,
                                TRmsThreshold const threshold) const -> void {
    auto stddevCheckLambda = [&](const uint64_t id) {
      // check if measured RMS exceeds threshold
      if (initPedestalMap[0][id].stddev >
          threshold * threshold * Config::MOVING_STAT_WINDOW_SIZE)
        mask->data[id] = false;
    };

    // execute double loop to take advantage of SIMD
    forEach(getLinearIdx(acc), getLinearElementExtent(acc), Config::MAPSIZE,
            stddevCheckLambda);
  }
};
