#pragma once

#include "../AlpakaHelper.hpp"
#include "ForEach.hpp"
#include <alpaka/alpaka.hpp>

template <typename Config> struct MaxValueKernel {
  ALPAKA_NO_HOST_ACC_WARNING

  template <typename TAcc, typename TEnergyMap, typename TEnergyValue,
            typename TNumFrames>
  ALPAKA_FN_ACC auto operator()(TAcc const &acc, TEnergyMap const *const energy,
                                TEnergyValue *const maxValues,
                                TNumFrames const numFrames) const -> void {
    for (TNumFrames i = 0; i < numFrames; ++i) {
      auto maxValueLambda = [&](const uint64_t id) {
        alpakaAtomicMax(acc, &maxValues[i], energy[i].data[id]);
      };

      // iterate over all elements in the thread
      forEach(getLinearIdx(acc), getLinearElementExtent(acc), Config::MAPSIZE,
              maxValueLambda);
    }
  }
};
