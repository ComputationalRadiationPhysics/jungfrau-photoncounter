#include "../Config.hpp"

struct ClusterFinderKernel {
    template <typename TAcc,
              typename TDetectorData,
              typename TGainMap,
              typename TPedestalMap,
              typename TGainStageMap,
              typename TEnergyMap,
              typename TClusterArray,
              typename TNumFrames,
              typename TClusterSize,
              typename TNumStdDevs
              >
    ALPAKA_FN_ACC auto operator()(TAcc const& acc,
                                  TDetectorData const* const detectorData,
                                  TGainMap const* const gainMaps,
                                  TPedestalMap* const pedestalMaps,
                                  TGainStageMap const* const gainStageMaps,
                                  TEnergyMap const* const energyMaps,
                                  TClusterArray* const clusterArray,
                                  TNumFrames const numFrames,
                                  TClusterSize const n,
                                  TNumStdDevs const c = 5,
                                  ) const -> void
    {
        auto const globalThreadIdx =
            alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        auto const globalThreadExtent =
            alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

        auto const linearizedGlobalThreadIdx =
            alpaka::idx::mapIdx<1u>(globalThreadIdx, globalThreadExtent);

        auto id = linearizedGlobalThreadIdx[0u];

        for (TNumFrames i = 0; i < numFrames; ++i) {
            auto adc = getAdc(detectorData[i].data[id]);
            const auto& gainStage = gainStageMaps[i].data[id];
            float sum;
            decltype(id) max;
            const auto gainStage = gainStageMaps[i][id];
            const auto& energy = energyMaps[i].data[id];
            const auto& pedestal = pedestalMaps[gainStage][id].mean;
            const auto& stddev = pedestalMaps[gainStage][id].stddev;
            if (indexQualifiesAsClusterCenter(id)) {
                findClusterSumAndMax(energyMaps[i], id, n, sum, max);
                // check cluster conditions
                if ((energy > c * stddev || sum > n * c * stddev) && id == max) {
                    auto& cluster = getClusterBuffer(acc, clusterArray);
                    copyCluster(energyMaps[i], id, n, cluster);
                }
                // check dark pixel condition
                else if (-c * stddev <= energy && c * stddev >= energy) {
                    updatePedestal(adc, pedestalMaps[gainStage][id]);
                }
            }
        }
    }
};
