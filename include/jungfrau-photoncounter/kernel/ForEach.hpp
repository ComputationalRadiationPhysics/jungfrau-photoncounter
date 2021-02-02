#pragma once
#include "helpers.hpp"

template <typename TFunctor, typename... TArgs>
ALPAKA_FN_ACC ALPAKA_FN_INLINE void
forEach(uint32_t workerIdx, uint32_t workerSize, uint32_t domainSize,
        TFunctor &&functor, TArgs &&... args) {
  // iterate over whole extent
  uint32_t const iterationExtent =
      std::min(workerSize, domainSize - workerIdx * workerSize);

#pragma omp simd
  for (uint32_t i = 0u; i < iterationExtent; ++i) {
    uint32_t const localIdx = workerIdx * workerSize + i;

    // call functor
    functor(localIdx, std::forward<TArgs>(args)...);
  }
}
