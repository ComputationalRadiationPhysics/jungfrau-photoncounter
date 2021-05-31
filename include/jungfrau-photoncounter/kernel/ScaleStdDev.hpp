#pragma once
#include "ForEach.hpp"
#include "helpers.hpp"

template <typename Config> struct ScaleStdDevKernel {
  template <typename TAcc, typename TInitPedestalMap,
            typename TNumStdDevs = int>
  ALPAKA_FN_ACC auto operator()(TAcc const &acc,
                                TInitPedestalMap *const initPedestalMap,
                                TNumStdDevs const c = Config::C) const -> void {
    constexpr auto PEDEMAPS = Config::PEDEMAPS;
    uint32_t workerSize = getLinearElementExtent(acc);

    // execute double loop to take advantage of SIMD
    forEach(getLinearIdx(acc), workerSize, Config::MAPSIZE,
            [&](const uint64_t id) {
              for (uint8_t i = 0; i < PEDEMAPS; ++i) {
                initPedestalMap[i][id].stddev *= c;
              }
            });
  }
};
