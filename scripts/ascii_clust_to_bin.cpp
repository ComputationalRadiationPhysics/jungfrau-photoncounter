#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <vector>

constexpr std::size_t CLUSTER_SIZE = 3;
constexpr std::size_t MAX_CLUSTER_NUM = 25000000;

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

ClusterArray readClusters(const char* path)
{
    ClusterArray clusters;
    Cluster* clusterPtr = clusters.clusters.get();

    std::cout << "Reading " << path << " ...\n";

    std::ifstream clusterFile(path);

    clusterFile >> clusters.used;

    std::size_t i = 0;
    std::size_t clusterCount = 0;
    // int32_t lastFrameNumber = -1;
    // Cluster* clusterPtr = nullptr;
    while (clusterFile.good()) {
        // write cluster information
        int32_t frameNumber;

        clusterFile >> frameNumber;
        // clusterFile.read(reinterpret_cast<char*>(&frameNumber),
        //              sizeof(int32_t));

        // if(lastFrameNumber != frameNumber) {
        //  if(!frames.empty())
        //    frames.rbegin()->clusters.used = i - 1;
        //  i = 0;
        //  frames.emplace_back(frameNumber);
        //  clusterPtr = frames.rbegin()->clusters.clusters.get();
        //  lastFrameNumber = frameNumber;
        //}

        clusterPtr[i].frameNumber = frameNumber & 0xFFFFFFFF;

        clusterFile >> clusterPtr[i].x;
        clusterFile >> clusterPtr[i].y;
        // clusterFile.read(reinterpret_cast<char*>(&clusterPtr[i].x),
        //                 sizeof(clusterPtr[i].x));
        // clusterFile.read(reinterpret_cast<char*>(&clusterPtr[i].y),
        //                 sizeof(clusterPtr[i].y));

        // write cluster
        for (uint8_t y = 0; y < CLUSTER_SIZE; ++y) {
            for (uint8_t x = 0; x < CLUSTER_SIZE; ++x) {
                clusterFile >> clusterPtr[i].data[x + y * CLUSTER_SIZE];
                //  clusterFile.read(
                //      reinterpret_cast<char*>(
                //          &clusterPtr[i].data[x + y * CLUSTER_SIZE]),
                //      sizeof(clusterPtr[i].data[x + y * CLUSTER_SIZE]));
            }
        }
        ++i;
        ++clusterCount;

        if (i >= MAX_CLUSTER_NUM) {
            std::cerr << "Cluster overflow!\n";
            exit(EXIT_FAILURE);
        }
    }

    if (0 == i)
        std::cerr << "Warning: No clusters loaded\n";

    clusterFile.close();
    
    std::cout << "Read " << clusterCount - 1 << " clusters.\n";

    return clusters;
}

void saveClustersBin(const std::string& path, const ClusterArray& clusters)
{
    std::ofstream clusterFile(path.c_str(), std::ios::binary);
    Cluster* clusterPtr = clusters.clusters.get();

    std::cout << "writing " << clusters.used << " clusters to " << path << "\n";

    for (uint64_t i = 0; i < clusters.used; ++i) {
        // write cluster information
        int32_t frameNumber = clusterPtr[i].frameNumber & 0xFFFFFFFF;
        clusterFile.write(reinterpret_cast<char*>(&frameNumber),
                          sizeof(int32_t));
        clusterFile.write(reinterpret_cast<char*>(&clusterPtr[i].x),
                          sizeof(clusterPtr[i].x));
        clusterFile.write(reinterpret_cast<char*>(&clusterPtr[i].y),
                          sizeof(clusterPtr[i].y));

        // write cluster
        for (uint8_t y = 0; y < CLUSTER_SIZE; ++y) {
            for (uint8_t x = 0; x < CLUSTER_SIZE; ++x) {
                clusterFile.write(
                    reinterpret_cast<char*>(
                        &clusterPtr[i].data[x + y * CLUSTER_SIZE]),
                    sizeof(clusterPtr[i].data[x + y * CLUSTER_SIZE]));
            }
        }
    }

    clusterFile.flush();
    clusterFile.close();
}

int main(int argc, char* argv[])
{
    if (argc != 3)
        exit(EXIT_FAILURE);

    char* input(argv[1]);
    std::string output(argv[2]);

    const ClusterArray clusters = readClusters(input);
    saveClustersBin(output, clusters);

    return EXIT_SUCCESS;
}
