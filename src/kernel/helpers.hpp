#pragma once
#include "../Config.hpp"
#include <alpaka/alpaka.hpp>

template <typename TDataword>
ALPAKA_FN_ACC ALPAKA_FN_INLINE auto getAdc(TDataword dataword) -> uint16_t
{
    return dataword & 0x3fff;
}

template <typename TDataword>
ALPAKA_FN_ACC ALPAKA_FN_INLINE auto getGainStage(TDataword dataword) -> uint8_t
{
    auto g = (dataword & 0xc000) >> 14;
    // map gain stages 0, 1, 3 to 0, 1, 2
    if (g == 3)
        g = 2;
    return g;
}

template <typename TInputMap, typename TOutputMap>
ALPAKA_FN_ACC ALPAKA_FN_INLINE auto copyFrameHeader(TInputMap const& src, TOutputMap& dst) -> void
{
    dst.header = src.header;
}

template <typename TAcc, typename TAdcValue, typename TPedestal>
ALPAKA_FN_ACC ALPAKA_FN_INLINE auto
updatePedestal(const TAcc& acc, TAdcValue const adc, TPedestal& pedestal) -> void
{
    // online algorithm for variance by Welford
    auto& count = pedestal.count;
    auto& mean = pedestal.mean;
    auto& m2 = pedestal.m2;
    auto& var = pedestal.variance; // sample variance
    auto& stddev = pedestal.stddev;

    ++count;
    float delta = adc - mean;
    mean += delta / count;
    float delta2 = adc - mean;
    m2 += delta * delta2;
    var = m2 / (count - 1);
    stddev = alpaka::math::sqrt(acc, var);
}

template <typename TThreadIndex>
ALPAKA_FN_ACC ALPAKA_FN_INLINE auto indexQualifiesAsClusterCenter(TThreadIndex id) -> bool
{
    constexpr auto n = CLUSTER_SIZE;
    return (id % DIMX >= n / 2 && id % DIMX <= DIMX - (n + 1) / 2 &&
            id / DIMX >= n / 2 && id / DIMX <= DIMY - (n + 1) / 2);
}

template <typename TMap, typename TThreadIndex, typename TSum>
ALPAKA_FN_ACC ALPAKA_FN_INLINE auto findClusterSumAndMax(TMap const& map,
                                        TThreadIndex const id,
                                        TSum& sum,
                                        TThreadIndex& max) -> void
{
    TThreadIndex it = 0;
    max = 0;
    constexpr auto n = CLUSTER_SIZE;
    for (int y = -n / 2; y < (n + 1) / 2; ++y) {
        for (int x = -n / 2; x < (n + 1) / 2; ++x) {
            it = id + y * DIMX + x;
            if (map[max] < map[it])
                max = it;
            sum += map[it];
        }
    }
}

template <typename TAcc, typename TClusterArray>
ALPAKA_FN_ACC ALPAKA_FN_INLINE auto getClusterBuffer(TAcc const& acc,
                                     TClusterArray* const clusterArray) -> TClusterArray&
{
    // get next free index of buffer atomically
    auto i = alpaka::atomic::atomicOp<alpaka::atomic::op::Add>(
        acc, &clusterArray->size, 1);
    return clusterArray->clusters[i];
}

template <typename TMap, typename TThreadIndex, typename TCluster>
ALPAKA_FN_ACC ALPAKA_FN_INLINE auto
copyCluster(TMap const& map, TThreadIndex const id, TCluster& cluster) -> void
{
    constexpr auto n = CLUSTER_SIZE;
    TThreadIndex it;
    TThreadIndex i = 0;
    cluster.x = id % DIMX;
    cluster.y = id / DIMX;
    cluster.frameNumber = map.header.frameNumber;
    for (int y = -n / 2; y < (n + 1) / 2; ++y) {
        for (int x = -n / 2; x < (n + 1) / 2; ++x) {
            it = id + y * DIMX + x;
            cluster.data[i++] = map[it];
        }
    }
}

template <typename TAcc,
          typename TDetectorData,
          typename TGainMap,
          typename TPedestalMap,
          typename TGainStageMap,
          typename TEnergyMap,
          typename TNumFrames,
          typename TMaskMap,
          typename TIndex
          >
ALPAKA_FN_ACC ALPAKA_FN_INLINE auto
processInput(TAcc const& acc,
              TDetectorData const& detectorData,
              TGainMap const* const gainMaps,
              TPedestalMap* const pedestalMaps,
              TGainStageMap& gainStageMap,
              TEnergyMap& energyMap,
              TMaskMap* const mask,
              TIndex const id) -> void
{
    // use masks to check whether the channel is valid or masked out
    bool isValid = mask->data[id];

    auto dataword = detectorData.data[id];
    auto adc = getAdc(dataword);
    
    auto& gainStage = gainStageMap.data[id];
    gainStage = getGainStage(dataword);

    // first thread copies frame header to output maps
    if (id == 0) {
        copyFrameHeader(detectorData, energyMap);
        copyFrameHeader(detectorData, gainStageMap);
    }

    const auto& pedestal = pedestalMaps[gainStage][id].mean;
    const auto& gain = gainMaps[gainStage][id];

    // calculate energy of current channel
    auto& energy = energyMap.data[id];
    energy = (adc - pedestal) / gain;

    // set energy to zero if masked out
    if (!isValid)
        energy = 0;
}
