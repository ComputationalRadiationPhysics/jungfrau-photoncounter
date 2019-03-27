#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>

template <std::size_t N_>
struct Cluster {
    static constexpr std::size_t N = N_;
    std::uint64_t frameNumber;
    std::int16_t x;
    std::int16_t y;
    std::int32_t data[N * N];
};

struct FrameHeader {
    std::uint64_t frameNumber;
    std::uint64_t bunchId;
};

template <typename T, std::size_t WIDTH = 1024, std::size_t HEIGHT = 512>
struct Frame {
    FrameHeader header;
    T data[WIDTH * HEIGHT];
    static constexpr std::size_t width = WIDTH;
    static constexpr std::size_t height = HEIGHT;
};

using InputData = Frame<std::uint16_t>;

struct Point {
    std::int16_t x;
    std::int16_t y;
};

Point partition(std::int16_t width, std::int16_t height, int n) {
    if (n == 0)
        return {width, height};
    else if (width > height)
        return partition(width / 2, height, n - 1);
    else
        return partition(width, height / 2, n - 1);
}

std::vector<Point> makeClusterCenters(std::int16_t width, std::int16_t height, int n) {
    int partitionRuns = std::ceil(std::log2(n));
    std::vector<Point> centers;
    Point dims = partition(width, height, partitionRuns);
    Point p;
    for (int x = 0; x < width; x += dims.x) {
        for (int y = 0; y < height; y += dims.y) {
            p.x = x + dims.x / 2;
            p.y = y + dims.y / 2;
            centers.push_back(p);
        }
    }
    centers.resize(n);
    return centers;
}

template <class TFrame, class TValue, class Cluster>
void addClusters(TFrame &frame, const std::vector<Point> &centers, std::vector<Cluster> &clusters,
                 TValue value) {
    static constexpr auto N = Cluster::N;
    const auto upper = N / 2 + (N % 2 > 0);
    const auto lower = N / 2;
    Cluster cluster;
    cluster.frameNumber = frame.header.frameNumber;
    std::fill(std::begin(cluster.data), std::end(cluster.data), value);
    for (auto &p : centers) {
        cluster.x = p.x;
        cluster.y = p.y;
        for (int l = p.x - lower; l <= p.x + upper - 1; ++l) {
            for (int m = p.y - lower; m <= p.y + upper - 1; ++m) {
                frame.data[m * frame.width + l] = value;
            }
        }
        clusters.push_back(cluster);
    }
}

template <class TFrame, class TValue>
void fillFrame(TFrame &frame, TValue value) {
    std::fill(std::begin(frame.data), std::end(frame.data), value);
}

int main() {
    Frame<std::uint16_t, 1024, 512> frame;
    std::fill(std::begin(frame.data), std::end(frame.data), 0);
    std::vector<Cluster<10>> clusters;
    auto centers = makeClusterCenters(frame.width, frame.height, 32);
    addClusters(frame, centers, clusters, 255);
    std::ofstream output{"/tmp/jf/output.txt"};
    std::for_each(std::begin(frame.data), std::end(frame.data),
                  [&output](auto x) { output << x << " "; });
    return 0;
}
