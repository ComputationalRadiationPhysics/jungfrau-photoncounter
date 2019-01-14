#pragma once
#include "../Config.hpp"
#include "helpers.hpp"

struct ClusterFinderKernel {
    template <typename TAcc,
              typename TDetectorData,
              typename TGainMap,
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
                                  TPedestalMap* const pedestalMaps,
                                  TGainStageMap* const gainStageMaps,
                                  TEnergyMap* const energyMaps,
                                  TClusterArray* const clusterArray,
                                  TNumClusters* const numClusters,
                                  TMask* const mask,
                                  TNumFrames const numFrames,
                                  TCurrentFrame const currentFrame,
                                  TNumStdDevs const c = 5) const -> void
    {
        auto const globalThreadIdx =
            alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        auto const globalThreadExtent =
            alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

        auto const linearizedGlobalThreadIdx =
            alpaka::idx::mapIdx<1u>(globalThreadIdx, globalThreadExtent);

        auto id = linearizedGlobalThreadIdx[0u];

        constexpr auto n = CLUSTER_SIZE;
        
        if (currentFrame) {                
            auto adc = getAdc(detectorData[currentFrame - 1].data[id]);
            const auto& gainStage = gainStageMaps[currentFrame - 1].data[id];
            float sum;
            decltype(id) max;
            const auto& energy = energyMaps[currentFrame - 1].data[id];
            const auto& stddev = pedestalMaps[gainStage][id].stddev;
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
                else if (-c * stddev <= energy && c * stddev >= energy) {
                    updatePedestal(acc, adc, pedestalMaps[gainStage][id]);
                }
            }
        }

        if (currentFrame < numFrames)
            processInput(acc,
                         detectorData[currentFrame],
                         gainMaps,
                         pedestalMaps,
                         gainStageMaps[currentFrame],
                         energyMaps[currentFrame],
                         mask,
                         id);
    }
};
