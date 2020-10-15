#pragma once
#include "Config.hpp"
#include "ForEach.hpp"
#include "helpers.hpp"

// original implementation
template <typename Config, typename AccConfig> struct ClusterFinderKernel {
  template <typename TAcc, typename TDetectorData, typename TGainMap,
            typename TInitPedestalMap, typename TPedestalMap,
            typename TGainStageMap, typename TEnergyMap, typename TClusterArray,
            typename TNumClusters, typename TMask, typename TNumFrames,
            typename TCurrentFrame, typename TNumStdDevs = int>
  ALPAKA_FN_ACC auto operator()(
      TAcc const &acc, TDetectorData const *const detectorData,
      TGainMap const *const gainMaps, TInitPedestalMap *const initPedestalMaps,
      TPedestalMap *const pedestalMaps, TGainStageMap *const gainStageMaps,
      TEnergyMap *const energyMaps, TClusterArray *const clusterArray,
      TNumClusters *const numClusters, TMask *const mask,
      TNumFrames const numFrames, TCurrentFrame const currentFrame,
      bool pedestalFallback, TNumStdDevs const c = Config::C) const -> void {

    constexpr auto n = Config::CLUSTER_SIZE;

    auto clusterLambda = [&](const uint64_t id) {
      if (currentFrame) {
        auto adc = getAdc(detectorData[currentFrame - 1].data[id]);
        const auto &gainStage = gainStageMaps[currentFrame - 1].data[id];
        float sum;
        uint64_t max;
        const auto &energy = energyMaps[currentFrame - 1].data[id];
        const auto &stddev = initPedestalMaps[gainStage][id].stddev;
        if (indexQualifiesAsClusterCenter<Config>(id)) {
          findClusterSumAndMax<Config>(energyMaps[currentFrame - 1].data, id,
                                       sum, max);
          // check cluster conditions
          if ((energy > c * stddev || sum > n * c * stddev) && id == max) {
            auto &cluster = getClusterBuffer(acc, clusterArray, numClusters);
            copyCluster<Config>(energyMaps[currentFrame - 1], id, cluster);
          }
          // check dark pixel condition
          else if (-c * stddev <= energy && c * stddev >= energy &&
                   !pedestalFallback) {
            updatePedestal(adc, Config::MOVING_STAT_WINDOW_SIZE,
                           pedestalMaps[gainStage][id]);
          }
        }
      }

      if (currentFrame < numFrames) {
        // first thread copies frame header to output
        if (getLinearIdx(acc) == 0) {
          copyFrameHeader(detectorData[currentFrame], energyMaps[currentFrame]);
          copyFrameHeader(detectorData[currentFrame],
                          gainStageMaps[currentFrame]);
        }

        processInput(acc, detectorData[currentFrame], gainMaps, pedestalMaps,
                     initPedestalMaps, gainStageMaps[currentFrame],
                     energyMaps[currentFrame], mask, id, pedestalFallback);
      }
    };

    // execute double loop to take advantage of SIMD
    forEach(getLinearIdx(acc), AccConfig::elementsPerThread, Config::MAPSIZE,
            clusterLambda);
  }
};
