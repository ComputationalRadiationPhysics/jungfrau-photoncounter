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
              typename TNumStdDevs
              >
    ALPAKA_FN_ACC auto operator()(TAcc const& acc,
                                  TDetectorData const* const detectorData,
                                  TGainMap const* const gainMaps,
                                  TPedestalMap* const pedestalMaps,
                                  TGainStageMap const* const gainStageMaps,
                                  TEnergyMap const* const energyMaps,
                                  TClusterArray* const clusterArray,
                                  TNumClusters* const numClusters,
                                  TMask* const mask,
                                  TNumFrames const numFrames,
                                  TNumStdDevs const c = 5
                                  ) const -> void
    {
        auto const globalThreadIdx =
            alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        auto const globalThreadExtent =
            alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

        auto const linearizedGlobalThreadIdx =
            alpaka::idx::mapIdx<1u>(globalThreadIdx, globalThreadExtent);

        auto id = linearizedGlobalThreadIdx[0u];

        constexpr auto n = CLUSTER_SIZE;
        for (TNumFrames i = 0; i < numFrames; ++i) {
            processInput(acc,
                    detectorData[i],
                    gainMaps,
                    pedestalMaps,
                    gainStageMaps[i],
                    energyMaps[i],
                    mask,
                    id);
            
            auto adc = getAdc(detectorData[i].data[id]);
            const auto& gainStage = gainStageMaps[i].data[id];
            float sum;
            decltype(id) max;
            const auto& energy = energyMaps[i].data[id];
            const auto& pedestal = pedestalMaps[gainStage][id].mean;
            const auto& stddev = pedestalMaps[gainStage][id].stddev;
            if (indexQualifiesAsClusterCenter(id)) {
                findClusterSumAndMax(energyMaps[i], id, sum, max);
                // check cluster conditions
                if ((energy > c * stddev || sum > n * c * stddev) 
                        && id == max) {
                  auto& cluster = getClusterBuffer(acc, clusterArray, numClusters);
                    copyCluster(energyMaps[i], id, cluster);
                }
                // check dark pixel condition
                else if (-c * stddev <= energy && c * stddev >= energy) {
                    updatePedestal(acc, adc, pedestalMaps[gainStage][id]);
                }
            }
        }
    }
};
