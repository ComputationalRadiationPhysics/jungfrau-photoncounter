#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// detector value struct
struct DetectorValue {
  uint16_t adc : 14;
  uint8_t gainStage : 2;
};

// frame header struct
struct FrameHeader {
  std::uint64_t frameNumber;
  std::uint64_t bunchId;
};

// frame struct
struct Frame {
  static constexpr unsigned int width = 1024;
  static constexpr unsigned int height = 512;
  FrameHeader header;
  DetectorValue data[width * height];
};

// structure for points
struct Point {
  std::uint16_t x;
  std::uint16_t y;
};

// structure for clusters
template <std::size_t N_> struct Cluster {
  static constexpr std::size_t N = N_;
  std::uint64_t frameNumber;
  std::uint16_t x;
  std::uint16_t y;
  std::int32_t data[N * N];
};

// calibration data calibrator
void generateCalibration(std::string path) {
  // open output file
  std::ofstream out(path.c_str(), std::ios::binary);
  if (!out.is_open()) {
    std::cerr << "Couldn't open pedestal output file!\n";
    return;
  }

  // iterate over gain stages
  for (uint8_t gainStage = 0; gainStage < 3; ++gainStage) {
    // convert gain stage
    if (gainStage == 2)
      gainStage = 3;

    // iterate over frames for each gain stage
    for (uint64_t frame = 0; frame < ((gainStage == 3) ? 999 : 1000); ++frame) {
      // write frame number and bunch ID
      uint64_t bunchID = 0;
      out.write(reinterpret_cast<char *>(&frame), sizeof(frame));
      out.write(reinterpret_cast<char *>(&bunchID), sizeof(bunchID));

      // iterate over rows
      for (unsigned int y = 0; y < 1024; ++y) {
        // iterate over cells
        for (unsigned int x = 0; x < 512; ++x) {
          // write value
          DetectorValue value{static_cast<uint16_t>(1000 * (gainStage + 1)),
                              gainStage};
          out.write(reinterpret_cast<char *>(&value), sizeof(value));
        }
      }
    }
  }

  // close file
  out.flush();
  out.close();
}

// test data generator for gain stage 0
void generateMainG0(std::string path,
                    unsigned int frameCount = 10000) { // open output file
  std::ofstream out(path.c_str(), std::ios::binary);
  if (!out.is_open()) {
    std::cerr << "Couldn't open pedestal output file!\n";
    return;
  }

  // iterate over frames
  for (uint64_t frame = 0; frame < frameCount; ++frame) {
    // write frame number and bunch ID
    uint64_t bunchID = 0;
    out.write(reinterpret_cast<char *>(&frame), sizeof(frame));
    out.write(reinterpret_cast<char *>(&bunchID), sizeof(bunchID));

    // iterate over rows
    for (unsigned int y = 0; y < 1024; ++y) {
      // iterate over cells
      for (unsigned int x = 0; x < 512; ++x) {
        // write value
        DetectorValue value{static_cast<uint16_t>(1000), 0u};
        out.write(reinterpret_cast<char *>(&value), sizeof(value));
      }
    }
  }

  // close file
  out.flush();
  out.close();
}

// test data generator for gain stage 1 and 3
void generateMainG13(std::string path, unsigned int frameCount = 10000) {
  // open output file
  std::ofstream out(path.c_str(), std::ios::binary);
  if (!out.is_open()) {
    std::cerr << "Couldn't open pedestal output file!\n";
    return;
  }

  // iterate over frames
  for (uint64_t frame = 0; frame < frameCount; ++frame) {
    // write frame number and bunch ID
    uint64_t bunchID = 0;
    out.write(reinterpret_cast<char *>(&frame), sizeof(frame));
    out.write(reinterpret_cast<char *>(&bunchID), sizeof(bunchID));

    // iterate over rows
    for (unsigned int y = 0; y < 1024; ++y) {
      // iterate over cells
      for (unsigned int x = 0; x < 512; ++x) {
        // set the first half to gain stage 1 and the other half to gain stage 3
        uint8_t gainStage = ((frame < frameCount / 2) ? 1u : 3u);

        // write value
        DetectorValue value{static_cast<uint16_t>(1000 * (gainStage + 1)),
                            gainStage};
        out.write(reinterpret_cast<char *>(&value), sizeof(value));
      }
    }
  }

  // close file
  out.flush();
  out.close();
}

// function to partition the frame
Point partition(std::uint16_t width, std::uint16_t height, int n) {
  if (n == 0)
    return {width, height};
  else if (width > height)
    return partition(width / 2, height, n - 1);
  else
    return partition(width, height / 2, n - 1);
}

// function to generate cluster centers
template <unsigned N>
std::vector<Point> makeClusterCenters(std::uint16_t width, std::uint16_t height,
                                      unsigned int n) {
  if (n <= 0)
    return std::vector<Point>();

  // calculate partition in which a cluster is located
  int partitionRuns{static_cast<int>(std::ceil(std::log2(n)))};
  std::vector<Point> centers;
  Point dims = partition(width, height, partitionRuns);

  // created cluster in center of each partition
  Point p;
  for (int x = 0; x < width; x += dims.x) {
    for (int y = 0; y < height; y += dims.y) {
      unsigned maxX = x + dims.x / 2 + N / 2;
      unsigned maxY = y + dims.y / 2 + N / 2;

      // note: this check might introduce small offsets of the desired number of
      // clusters in edge cases
      if (centers.size() < n && maxX < width && maxY < height) {
        p.x = static_cast<uint16_t>(x + dims.x / 2);
        p.y = static_cast<uint16_t>(y + dims.y / 2);
        centers.push_back(p);
      }
    }
  }
  centers.resize(n);
  return centers;
}

// function to add clusters
template <unsigned TCLusterSize>
void addClusters(Frame &frame, const std::vector<Point> &centers,
                 std::vector<Cluster<TCLusterSize>> &clusters) {
    // init variables
  static constexpr auto N = Cluster<TCLusterSize>::N;
  const auto upper = N / 2 + (N % 2 > 0);
  const auto lower = N / 2;
  Cluster<TCLusterSize> cluster;
  cluster.frameNumber = frame.header.frameNumber;

  // define values to fill clusters with
  DetectorValue value{1010u, 0u};
  DetectorValue centerValue{1012u, 0u};

  // clear the frame
  std::fill(std::begin(cluster.data), std::end(cluster.data), 1010u);
  cluster.data[(N / 2) * N + N / 2] = 1012u;

  // insert clusters around centers
  for (auto &p : centers) {
    cluster.x = p.x;
    cluster.y = p.y;
    for (unsigned int l = p.x - lower; l <= p.x + upper - 1; ++l) {
      for (unsigned int m = p.y - lower; m <= p.y + upper - 1; ++m) {
        if (l == p.x && m == p.y)
          frame.data[m * frame.width + l] = value;
        else
          frame.data[m * frame.width + l] = centerValue;
      }
    }
    clusters.push_back(cluster);
  }
}

// cluster test data generator
template <unsigned TCLusterSize>
void generateCluster(std::string path, float clusterAmount,
                     unsigned int frameCount = 10000) {
  // open output file
  std::ofstream out(path.c_str(), std::ios::binary);
  if (!out.is_open()) {
    std::cerr << "Couldn't open pedestal output file!\n";
    return;
  }

  // iterate over frames
  for (uint64_t frameNumber = 0; frameNumber < frameCount; ++frameNumber) {
    // clear frame
    Frame frame;
    DetectorValue empty{1000u, 0u};
    std::fill(std::begin(frame.data), std::end(frame.data), empty);
    unsigned int clusterCount =
        static_cast<unsigned int>(1024 * 512 * clusterAmount);

    // add clusters
    std::vector<Cluster<TCLusterSize>> clusters;
    auto centers = makeClusterCenters<TCLusterSize>(frame.width, frame.height,
                                                    clusterCount);
    addClusters<TCLusterSize>(frame, centers, clusters);

    // write ouptut
    out.write(reinterpret_cast<char *>(&frame), sizeof(frame));
  }

  // close file
  out.flush();
  out.close();
}

int main() {
  generateCalibration("/bigdata/hplsim/production/jungfrau-photoncounter/data_pool/synthetic/pede.bin");
  generateMainG0("/bigdata/hplsim/production/jungfrau-photoncounter/data_pool/synthetic/g0.bin", 10000);
  generateMainG13("/bigdata/hplsim/production/jungfrau-photoncounter/data_pool/synthetic/g13.bin", 10000);
  generateCluster<3>("/bigdata/hplsim/production/jungfrau-photoncounter/data_pool/synthetic/cluster_0.bin", 0.f, 10000);
  generateCluster<3>("/bigdata/hplsim/production/jungfrau-photoncounter/data_pool/synthetic/cluster_4.bin", 0.04f, 10000);
  generateCluster<3>("/bigdata/hplsim/production/jungfrau-photoncounter/data_pool/synthetic/cluster_8.bin", 0.08f, 10000);
  return 0;
}
