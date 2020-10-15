#pragma once
#include "ForEach.hpp"
#include <alpaka/alpaka.hpp>

template <typename Config> struct MaxValueCopyKernel {
  ALPAKA_NO_HOST_ACC_WARNING

  template <typename TAcc, typename TEnergyMap, typename TEnergyValue,
            typename TNumFrames>
  ALPAKA_FN_ACC auto operator()(TAcc const &acc, TEnergyMap const *const source,
                                TEnergyValue *destination,
                                TNumFrames const &numFrames) const -> void {
    auto maxValueCopyLambda = [&](const uint64_t id) {
      if (id < numFrames) {
        destination[id] = source[id].data[0];
      }
    };

    // iterate over all elements in the thread
    forEach(getLinearIdx(acc), getLinearElementExtent(acc), Config::MAPSIZE,
            maxValueCopyLambda);
  }
};
