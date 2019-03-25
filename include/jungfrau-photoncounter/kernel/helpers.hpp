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
ALPAKA_FN_ACC ALPAKA_FN_INLINE auto copyFrameHeader(TInputMap const& src,
                                                    TOutputMap& dst) -> void
{
    dst.header = src.header;
}

template <typename TAcc, typename TAdcValue, typename TInitPedestal, typename TCountValue>
ALPAKA_FN_ACC ALPAKA_FN_INLINE auto
initPedestal(const TAcc& acc, TAdcValue const adc, TInitPedestal& initPedestal, TCountValue windowSize)
    -> void
{
    // online algorithm for variance by Welford
    auto& count = initPedestal.count;
    auto& mean = initPedestal.mean;
    auto& m = initPedestal.m;
    auto& m2 = initPedestal.m2;
    auto& stddev = initPedestal.stddev;

    ++count;
    // init
    if (count == 1) {
        m = adc;
        m2 = adc * adc;
    }
    // add
    else if (count <= windowSize) {
        m += adc;
        m2 += adc;
    }
    //push
    else {
      m += adc - m / windowSize;
      m2 += adc * adc - m2 / windowSize;
    }

    // calculate mean and stddev
    double currentWindowSize = (count > windowSize ? windowSize : count);
    mean = m / currentWindowSize;
    stddev = alpaka::math::sqrt(acc, m2 / currentWindowSize - mean * mean);
}

template <typename TAdcValue, typename TCountValue, typename TPedestal>
ALPAKA_FN_ACC ALPAKA_FN_INLINE auto updatePedestal(TAdcValue const adc,
                                                   TCountValue const count,
                                                   TPedestal& pedestal) -> void
{
    pedestal += (adc - pedestal) / count;
}

template <typename TThreadIndex>
ALPAKA_FN_ACC ALPAKA_FN_INLINE auto
indexQualifiesAsClusterCenter(TThreadIndex id) -> bool
{
    constexpr auto n = CLUSTER_SIZE;
    return (id % DIMX >= n / 2 && id % DIMX <= DIMX - (n + 1) / 2 &&
            id / DIMX >= n / 2 && id / DIMX <= DIMY - (n + 1) / 2);
}

template <typename TMap, typename TThreadIndex, typename TSum>
ALPAKA_FN_ACC ALPAKA_FN_INLINE auto findClusterSumAndMax(TMap const& map,
                                                         TThreadIndex const id,
                                                         TSum& sum,
                                                         TThreadIndex& max)
    -> void
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

template <typename TAcc, typename TClusterArray, typename TNumClusters>
ALPAKA_FN_ACC ALPAKA_FN_INLINE auto
getClusterBuffer(TAcc const& acc,
                 TClusterArray* const clusterArray,
                 TNumClusters* const numClusters) -> TClusterArray&
{
    // get next free index of buffer atomically
    auto i = alpaka::atomic::atomicOp<alpaka::atomic::op::Add>(
        acc, numClusters, static_cast<TNumClusters>(1));
    return clusterArray[i];
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
            auto tmp = map.data[it];
            cluster.data[i++] = tmp;
        }
    }
}

template <typename TAcc,
          typename TDetectorData,
          typename TGainMap,
          typename TPedestalMap,
          typename TInitPedestalMap,
          typename TGainStageMap,
          typename TEnergyMap,
          typename TMaskMap,
          typename TIndex>
ALPAKA_FN_ACC ALPAKA_FN_INLINE auto
processInput(TAcc const& acc,
             TDetectorData const& detectorData,
             TGainMap const* const gainMaps,
             TPedestalMap* const pedestalMaps,
             TInitPedestalMap* const initPedestalMaps,
             TGainStageMap& gainStageMap,
             TEnergyMap& energyMap,
             TMaskMap* const mask,
             TIndex const id,
             bool pedestalFallback) -> void
{
    // use masks to check whether the channel is valid or masked out
    bool isValid = !mask ? 1 : mask->data[id];

    auto dataword = detectorData.data[id];
    auto adc = getAdc(dataword);

    auto& gainStage = gainStageMap.data[id];
    gainStage = getGainStage(dataword);

    // first thread copies frame header to output maps
    if (id == 0) {
        copyFrameHeader(detectorData, energyMap);
        copyFrameHeader(detectorData, gainStageMap);
    }

    const auto& pedestal =
        (pedestalFallback ? initPedestalMaps[gainStage][id].mean
                          : pedestalMaps[gainStage][id]);
    const auto& gain = gainMaps[gainStage][id];

    // calculate energy of current channel
    auto& energy = energyMap.data[id];

    energy = (adc - pedestal) * gain;

    // set energy to zero if masked out
    if (!isValid)
        energy = 0;
}
