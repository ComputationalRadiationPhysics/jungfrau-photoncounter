#pragma once
#include "../Config.hpp"
#include "helpers.hpp"

struct ClusterFinderKernel {
    template <typename TAcc,
              typename TDetectorData,
              typename TGainMap,
              typename TInitPedestalMap,
              typename TPedestalMap,
              typename TGainStageMap,
              typename TEnergyMap,
              typename TClusterArray,
              typename TNumClusters,
              typename TMask,
              typename TNumFrames,
              typename TCurrentFrame,
              typename TNumStdDevs = int>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc,
                                  TDetectorData const* const detectorData,
                                  TGainMap const* const gainMaps,
                                  TInitPedestalMap* const initPedestalMaps,
                                  TPedestalMap* const pedestalMaps,
                                  TGainStageMap* const gainStageMaps,
                                  TEnergyMap* const energyMaps,
                                  TClusterArray* const clusterArray,
                                  TNumClusters* const numClusters,
                                  TMask* const mask,
                                  TNumFrames const numFrames,
                                  TCurrentFrame const currentFrame,
                                  bool pedestalFallback,
                                  TNumStdDevs const c = C) const -> void
    {
        auto id = getLinearIdx(acc);

        // check range
        if (id >= MAPSIZE)
            return;

        constexpr auto n = CLUSTER_SIZE;

        if (currentFrame) {
            auto adc = getAdc(detectorData[currentFrame - 1].data[id]);
            const auto& gainStage = gainStageMaps[currentFrame - 1].data[id];
            float sum;
            decltype(id) max;
            const auto& energy = energyMaps[currentFrame - 1].data[id];
            const auto& stddev = initPedestalMaps[gainStage][id].stddev;
            if (indexQualifiesAsClusterCenter(id)) {
                findClusterSumAndMax(
                    energyMaps[currentFrame - 1].data, id, sum, max);

                // check cluster conditions
                if ((energy > c * stddev || sum > n * c * stddev) &&
                    id == max) {
                    auto& cluster =
                        getClusterBuffer(acc, clusterArray, numClusters);
                    copyCluster(energyMaps[currentFrame - 1], id, cluster);
                }

                // check dark pixel condition
                else if (-c * stddev <= energy && c * stddev >= energy &&
                         !pedestalFallback) {
                    updatePedestal(adc,
                                   MOVING_STAT_WINDOW_SIZE,
                                   pedestalMaps[gainStage][id]);
                }
            }
        }

        if (currentFrame < numFrames)
            processInput(acc,
                         detectorData[currentFrame],
                         gainMaps,
                         pedestalMaps,
                         initPedestalMaps,
                         gainStageMaps[currentFrame],
                         energyMaps[currentFrame],
                         mask,
                         id,
                         pedestalFallback);
    }
};
