#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <vector>

constexpr std::size_t CLUSTER_SIZE = 3;
constexpr std::size_t MAX_CLUSTER_NUM = 10000000;

struct Cluster {
    std::uint64_t frameNumber;
    std::int16_t x;
    std::int16_t y;
    std::int32_t data[CLUSTER_SIZE * CLUSTER_SIZE];
};

struct ClusterArray {
    std::size_t used;
    std::unique_ptr<Cluster[]> clusters;

    explicit ClusterArray(std::size_t maxClusterCount = MAX_CLUSTER_NUM)
        : used(0), clusters(new Cluster[maxClusterCount])
    {
    }
};

struct ClusterFrame {
    explicit ClusterFrame(int32_t frameNumber) : frameNumber(frameNumber) {}
    ClusterArray clusters;
    int32_t frameNumber;
};

std::vector<ClusterFrame> readClusters(const char* path)
{
    std::vector<ClusterFrame> frames;

    std::cout << "Reading " << path << " ...\n";

    std::ifstream clusterFile(path, std::ios::binary);

    std::size_t i = 0;
    std::size_t clusterCount = 0;
    int32_t lastFrameNumber = -1;
    Cluster* clusterPtr = nullptr;
    while (clusterFile.good()) {
        // write cluster information
        int32_t frameNumber;

        clusterFile.read(reinterpret_cast<char*>(&frameNumber),
                         sizeof(int32_t));

        if (lastFrameNumber != frameNumber) {
            if (!frames.empty())
                frames.rbegin()->clusters.used = i - 1;
            i = 0;
            frames.emplace_back(frameNumber);
            clusterPtr = frames.rbegin()->clusters.clusters.get();
            lastFrameNumber = frameNumber;
        }

        clusterPtr[i].frameNumber = frameNumber & 0xFFFFFFFF;

        clusterFile.read(reinterpret_cast<char*>(&clusterPtr[i].x),
                         sizeof(clusterPtr[i].x));
        clusterFile.read(reinterpret_cast<char*>(&clusterPtr[i].y),
                         sizeof(clusterPtr[i].y));

        // write cluster
        for (uint8_t y = 0; y < CLUSTER_SIZE; ++y) {
            for (uint8_t x = 0; x < CLUSTER_SIZE; ++x) {
                clusterFile.read(
                    reinterpret_cast<char*>(
                        &clusterPtr[i].data[x + y * CLUSTER_SIZE]),
                    sizeof(clusterPtr[i].data[x + y * CLUSTER_SIZE]));
            }
        }
        ++i;
        ++clusterCount;

        if (i >= MAX_CLUSTER_NUM) {
            std::cerr << "Cluster overflow (over " << MAX_CLUSTER_NUM
                      << " clusters found)!\n";
            exit(EXIT_FAILURE);
        }
    }

    if (0 == i)
        std::cerr << "Warning: No clusters loaded\n";

    clusterFile.close();

    std::cout << "Read " << clusterCount - 1 << " clusters.\n";

    return frames;
}

bool checkFrameNumbers(std::vector<ClusterFrame>& clusters)
{
    std::cout << "Checking frame order ...\n";

    for (unsigned int i = 1; i < clusters.size(); ++i) {
        if (std::abs(clusters[i - 1].frameNumber - clusters[i].frameNumber) > 1) {
            std::cerr << "Framenumber out of order at position " << i << " / " << clusters.size() << " (" << static_cast<float>(i) / static_cast<float>(clusters.size()) * 100.0 <<"%): " << clusters[i - 1].frameNumber << " vs. " << clusters[i].frameNumber << "\n";
            return false;
        }
    }
    return true;
}

template <typename T, typename TCmp>
std::tuple<std::size_t, std::size_t> getOffset(T it1, T it2, TCmp cmp)
{
    std::size_t offset1 = 0;
    std::size_t offset2 = 0;
    if (cmp(it1->frameNumber, it2->frameNumber)) {
        while (cmp((++it1)->frameNumber, it2->frameNumber))
            ++offset1;
    } else if (cmp(it2->frameNumber, it1->frameNumber)) {
        while (cmp((++it2)->frameNumber, it1->frameNumber))
            ++offset2;
    }

    return std::make_tuple(offset1, offset2);
}

std::tuple<std::size_t, std::size_t, std::size_t, std::size_t>
selectCommonFrames(const std::vector<ClusterFrame>& v1,
                   const std::vector<ClusterFrame>& v2)
{
    const auto start =
        getOffset(v1.begin(), v2.begin(), std::less_equal<size_t>());
    const auto end =
        getOffset(v1.rbegin(), v2.rbegin(), std::greater<size_t>());
    std::size_t begin1 = std::get<0>(start);
    std::size_t end1 = std::get<0>(end);
    std::size_t begin2 = std::get<1>(start);
    std::size_t end2 = std::get<1>(end);
    
    return std::make_tuple(begin1, end1, begin2, end2);
}

int main(int argc, char* argv[])
{
    if (argc != 3)
        return EXIT_FAILURE;

    char* detector_path = argv[1];
    char* reference_path = argv[2];

    // read clusters
    std::vector<ClusterFrame> detector = readClusters(detector_path);
    std::vector<ClusterFrame> reference = readClusters(reference_path);

    // check frame numbers
    if (!checkFrameNumbers(detector)) {
        std::cerr << "Error: detector clusters not in order!\n";
        exit(-1);
    }

    if (!checkFrameNumbers(reference)) {
        std::cerr << "Error: reference clusters not in order!\n";
        exit(-1);
    }

    // extract offset information
    auto offsets = selectCommonFrames(detector, reference);
    size_t detector_begin = std::get<0>(offsets);
    size_t detector_end = detector.size() - std::get<1>(offsets);
    size_t reference_begin = std::get<2>(offsets);
    size_t reference_end = reference.size() - std::get<3>(offsets);

    std::size_t frameCount = reference_end - reference_begin;
    std::cout << "Processing " << frameCount << " common frames!\n";

    // calculate matches
    std::size_t frameIndex = 0;
    std::vector<std::size_t> exact_matches(frameCount),
        overlap_matches(frameCount), det_only(frameCount), ref_only(frameCount);
    for (; detector_begin < detector_end && reference_begin < reference_end;
         ++detector_begin, ++reference_begin) {
        const ClusterArray& referenceClusterArray =
            reference[reference_begin].clusters;
        const size_t& reference_size = referenceClusterArray.used;
        const Cluster* reference_clusters =
            referenceClusterArray.clusters.get();
        const ClusterArray& detectorClusterArray =
            detector[detector_begin].clusters;
        const size_t detector_size = detectorClusterArray.used;
        const Cluster* detector_clusters = detectorClusterArray.clusters.get();

        std::vector<bool> detector_matched(detector_size);
        std::fill(detector_matched.begin(), detector_matched.begin(), false);
        std::vector<bool> reference_matched(reference_size);
        std::fill(reference_matched.begin(), reference_matched.begin(), false);

        for (unsigned int det_idx = 0; det_idx < detector_size; ++det_idx) {
            for (unsigned int ref_idx = 0; ref_idx < reference_size;
                 ++ref_idx) {
                const int16_t& ref_x = reference_clusters[ref_idx].x;
                const int16_t& ref_y = reference_clusters[ref_idx].y;
                const int16_t& det_x = detector_clusters[det_idx].x;
                const int16_t& det_y = detector_clusters[det_idx].y;
                if (ref_x == det_x && ref_y == det_y)
                    ++exact_matches[frameIndex];
                if (std::pow(ref_x - det_x, 2) + std::pow(ref_y - det_y, 2) <
                    2) {
                    ++overlap_matches[frameIndex];
                    reference_matched[ref_idx] = true;
                    detector_matched[det_idx] = true;
                }
            }
        }

        det_only[frameIndex] =
            std::count(detector_matched.begin(), detector_matched.end(), false);
        ref_only[frameIndex] = std::count(
            reference_matched.begin(), reference_matched.end(), false);

        ++frameIndex;
    }

    std::ofstream stats("stats.txt");
    if (!stats.good())
        std::cerr << "Error: Couldn't write stats.txt file!\n";

    for (unsigned int frame = 0; frame < frameCount && stats.good(); ++frame) {
        stats << exact_matches[frame] << " " << overlap_matches[frame] << " "
              << det_only[frame] << " " << ref_only[frame] << "\n";
    }

    stats.flush();
    stats.close();

    return EXIT_SUCCESS;
}
