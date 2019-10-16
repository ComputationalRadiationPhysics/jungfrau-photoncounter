#pragma once
#include <alpaka/alpaka.hpp>

template <typename Config> struct MaxValueCopyKernel {
  ALPAKA_NO_HOST_ACC_WARNING

  template <typename TAcc, typename TEnergyMap, typename TEnergyValue,
            typename TNumFrames>
  ALPAKA_FN_ACC auto operator()(TAcc const &acc, TEnergyMap const *const source,
                                TEnergyValue *destination,
                                TNumFrames const &numFrames) const -> void {
    auto globalId = getLinearIdx(acc);
    auto elementsPerThread = getLinearElementExtent(acc);

    // iterate over all elements in the thread
    for (auto id = globalId * elementsPerThread;
         id < (globalId + 1) * elementsPerThread; ++id) {

      // check range
      if (id >= Config::MAPSIZE)
        break;

      if (id < numFrames) {
        destination[id] = source[id].data[0];
      }
    }
  }
};
