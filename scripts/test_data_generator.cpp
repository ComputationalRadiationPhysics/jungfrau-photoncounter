#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// detector value struct
struct DetectorValue {
  uint16_t adc : 14;
  uint8_t gainStage : 2;

  bool operator==(const DetectorValue &other) const {
    return adc == other.adc && gainStage == other.gainStage;
  }

  bool operator!=(const DetectorValue &other) const {
    return adc != other.adc || gainStage != other.gainStage;
  }
};

// define values to fill clusters with
constexpr DetectorValue value{1015u, 0u};
constexpr DetectorValue centerValue{1020u, 0u};
constexpr DetectorValue empty{1000u, 0u};

// frame header struct
struct FrameHeader {
  std::uint64_t frameNumber;
  std::uint64_t bunchId;

  //! @todo: remove this later; only for alignment performance tests
  //std::uint64_t dummy1;
  //std::uint64_t dummy2;
  //std::uint64_t dummy3;
  //std::uint64_t dummy4;
  //std::uint64_t dummy5;
  //std::uint64_t dummy6;
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
      for (unsigned int y = 0; y < Frame::width; ++y) {
        // iterate over cells
        for (unsigned int x = 0; x < Frame::height; ++x) {
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
    for (unsigned int y = 0; y < Frame::width; ++y) {
      // iterate over cells
      for (unsigned int x = 0; x < Frame::height; ++x) {
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
    for (unsigned int y = 0; y < Frame::width; ++y) {
      // iterate over cells
      for (unsigned int x = 0; x < Frame::height; ++x) {
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

bool isClose(Point p1, Point p2, int n) {
  std::int16_t diffx = std::max(p1.x, p2.x) - std::min(p1.x, p2.x);
  std::int16_t diffy = std::max(p1.y, p2.y) - std::min(p1.y, p2.y);

  std::int16_t diffx2 = diffx * diffx;
  std::int16_t diffy2 = diffy * diffy;

  std::int16_t diff = diffx2 + diffy2;
  std::int16_t cmp = (n * n + 3) / 4;

  return diff < cmp;
}

template <int N, typename TRng>
Point getValidRandomCenter(TRng rng, std::vector<Point> centers,
                           std::uint16_t width, std::uint16_t height,
                           bool allowOverlapping = true) {
  // calculate maximal number of possible cluster centers
  int n = (allowOverlapping ? N : N * 2);
  int pixelCount = (width - N + 1) * (height - N + 1);

  // check if all are already taken
  if (!pixelCount)
    return Point{0, 0};

  // select random cluster center
  int randIdx = rng() % pixelCount;

  // skip over all previous clusters
  for (const Point &c : centers) {
    // get position
    std::uint16_t x = N / 2 + randIdx % (width - N + 1);
    std::uint16_t y = N / 2 + randIdx / (width - N + 1);

    int cycleCounter = 0;

    // cluster lies entirely above
    if (c.y + (n + 1) / 2 < y) {
      // add full cluster size
      randIdx = (randIdx + n * n) % ((width - N + 1) * (height - N + 1));
    } else
      while (cycleCounter < pixelCount &&
             std::any_of(centers.begin(), centers.end(),
                         [&x, &y, &n](const Point &c) {
                           return isClose(Point{x, y}, c, n);
                         })) {
        // cluster lies partially above
        randIdx = (randIdx + 1) % ((width - N + 1) * (height - N + 1));

        x = N / 2 + randIdx % (width - N + 1);
        y = N / 2 + randIdx / (width - N + 1);

        ++cycleCounter;
      }
  }

  // calculate final position and return
  std::uint16_t finalx = N / 2 + randIdx % (width - N + 1);
  std::uint16_t finaly = N / 2 + randIdx / (width - N + 1);

  return Point{finalx, finaly};
}

// function to generate cluster centers
template <unsigned N>
std::vector<Point>
makeRandomClusterCenters(std::uint16_t width, std::uint16_t height,
                         unsigned int n, bool allowOverlapping = false) {
  // init
  if (n <= 0)
    return std::vector<Point>();
  std::vector<Point> centers;
  std::mt19937 gen;
  gen.seed(0);

  // pick one out randomly and remove cluster from possible centers
  for (unsigned int i = 0; i < n; ++i) {
    // take random point and store it
    Point p =
        getValidRandomCenter<N>(gen, centers, width, height, allowOverlapping);
    centers.emplace_back(p);
    std::sort(centers.begin(), centers.end(),
              [](const Point &p1, const Point &p2) {
                return p1.y * Frame::width + p1.x < p2.y * Frame::width + p2.x;
              });
  }

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
          frame.data[m * frame.width + l] = centerValue;
        else
          frame.data[m * frame.width + l] = value;
      }
    }
    clusters.push_back(cluster);
  }
}

// function to add clusters
template <unsigned TCLusterSize>
void addCluster(Frame &frame, Point center,
                std::vector<Cluster<TCLusterSize>> &clusters) {
  // init variables
  static constexpr auto N = Cluster<TCLusterSize>::N;
  const auto upper = N / 2 + (N % 2 > 0);
  const auto lower = N / 2;
  Cluster<TCLusterSize> cluster;
  cluster.frameNumber = frame.header.frameNumber;

  // clear the frame
  std::fill(std::begin(cluster.data), std::end(cluster.data), 1010u);
  cluster.data[(N / 2) * N + N / 2] = 1012u;

  // insert clusters around centers
  cluster.x = center.x;
  cluster.y = center.y;
  for (unsigned int l = center.x - lower; l <= center.x + upper - 1; ++l) {
    for (unsigned int m = center.y - lower; m <= center.y + upper - 1; ++m) {
      if (l == center.x && m == center.y)
        frame.data[m * frame.width + l] = centerValue;
      else
        frame.data[m * frame.width + l] = value;
    }
  }
  clusters.push_back(cluster);
}

template <unsigned TCLusterSize>
void generateRandomCluster(std::string path, float clusterAmount,
                           unsigned int frameCount = 10000,
                           bool allowOverlapping = false) {
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
    std::fill(std::begin(frame.data), std::end(frame.data), empty);
    unsigned int clusterCount =
        static_cast<unsigned int>(frame.width * frame.height * clusterAmount);

    // add clusters
    std::vector<Cluster<TCLusterSize>> clusters;
    auto centers = makeRandomClusterCenters<TCLusterSize>(
        frame.width, frame.height, clusterCount, allowOverlapping);
    addClusters<TCLusterSize>(frame, centers, clusters);

    // write ouptut
    out.write(reinterpret_cast<char *>(&frame), sizeof(frame));
  }

  // close file
  out.flush();
  out.close();
}

template <int N>
bool checkCluster(Point p, const Frame &f, bool allowOverlapping) {
  // check boundary condition
  if (p.x < N / 2 || p.x > Frame::width - N + (N % 2 > 0) || p.y < N / 2 ||
      p.y > Frame::height - N / 2 + (N % 2 > 0))
    return false;

  // iterate over cluster
  for (unsigned int y = p.y - N / 2; y < p.y + N / 2 + (N % 2 > 0); ++y) {
    for (unsigned int x = p.x - N / 2; x < p.x + N / 2 + (N % 2 > 0); ++x) {
      unsigned int idx = y * Frame::width + x;

      // skip empty pixels
      if (f.data[idx] == empty)
        continue;

      // other cluster center found
      else if (f.data[idx] == centerValue)
        return false;

      // part of other cluster found
      else if (f.data[idx] == value && !allowOverlapping)
        return false;
    }
  }

  // no other cluster found
  return true;
}

/*
class PRand {
public:
  PRand(int seed) : notRandom(seed) {}
  int operator()() { return ++notRandom; }

private:
  int notRandom;
};*/

class PRand {
public:
  PRand(int seed) : n(seed) {}
  int operator()() {
    n = (multiplier * n + increment) % modulo;
    return n;
  }

private:
  int n;

  static constexpr uint32_t modulo = 512 * 1024;
  static constexpr uint32_t multiplier = 5;
  static constexpr uint32_t increment = 1;
};

/*class PRand {
public:
  PRand(int seed) { gen.seed(seed); }
  int operator()() { return gen(); }

private:
  std::mt19937 gen;
};*/

template <unsigned TCLusterSize>
void generateRandomClusterFast(std::string path, float clusterAmount,
                               unsigned int frameCount = 10000,
                               bool allowOverlapping = false) {
  // open output file
  std::ofstream out(path.c_str(), std::ios::binary);
  if (!out.is_open()) {
    std::cerr << "Couldn't open pedestal output file!\n";
    return;
  }

  PRand gen(0);
  unsigned int clusterCount =
      static_cast<unsigned int>(Frame::width * Frame::height * clusterAmount);
  int pixelCount =
      (Frame::width - TCLusterSize + 1) * (Frame::height - TCLusterSize + 1);

  // iterate over frames
  for (uint64_t frameNumber = 0; frameNumber < frameCount; ++frameNumber) {
    // clear frame
    Frame frame;
    std::fill(std::begin(frame.data), std::end(frame.data), empty);
    std::vector<Cluster<TCLusterSize>> clusters;

    // add clusters
    for (unsigned int c = 0; c < clusterCount; ++c) {
      Point center;
      bool added_successfully = false;
      // try at most 1000 times to find a new cluster center
      for (int i = 0; i < 1024; ++i) {
        // randomly generate cluster center
        unsigned int idx = gen() % pixelCount;
        center.x = TCLusterSize / 2 + idx % (Frame::width - TCLusterSize + 1);
        center.y = TCLusterSize / 2 + idx / (Frame::width - TCLusterSize + 1);

        // check if the cluster center is valid
        if (checkCluster<TCLusterSize>(center, frame, allowOverlapping)) {
          addCluster<TCLusterSize>(frame, center, clusters);
          added_successfully = true;
          break;
        }
      }

      if (!added_successfully)
        std::cout << "Couldn't place cluster\n";
    }

    // write ouptut
    out.write(reinterpret_cast<char *>(&frame), sizeof(frame));
  }

  // close file
  out.flush();
  out.close();
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
    std::fill(std::begin(frame.data), std::end(frame.data), empty);
    unsigned int clusterCount =
        static_cast<unsigned int>(Frame::width * Frame::height * clusterAmount);

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
  // generateCalibration("/bigdata/hplsim/production/jungfrau-photoncounter/data_pool/synthetic/pede.bin");
  // generateMainG0("/bigdata/hplsim/production/jungfrau-photoncounter/data_pool/synthetic/g0.bin",
  // 10000);
  // generateMainG13("/bigdata/hplsim/production/jungfrau-photoncounter/data_pool/synthetic/g13.bin",
  // 10000);
  // generateCluster<3>("/bigdata/hplsim/production/jungfrau-photoncounter/data_pool/synthetic/cluster_0.bin",
  // 0.f, 10000);
  // generateCluster<3>("/bigdata/hplsim/production/jungfrau-photoncounter/data_pool/synthetic/cluster_4.bin",
  // 0.04f, 10000);
  // generateCluster<3>("/bigdata/hplsim/production/jungfrau-photoncounter/data_pool/synthetic/cluster_8.bin",
  // 0.08f, 10000);

  // generate random overlapping clusters
  /*generateRandomClusterFast<3>(
      "/bigdata/hplsim/production/jungfrau-photoncounter/data_pool/synthetic/"
      "random_clusters_overlapping/cluster_0.bin",
      0.f, 10000, true);
  generateRandomClusterFast<3>(
      "/bigdata/hplsim/production/jungfrau-photoncounter/data_pool/synthetic/"
      "random_clusters_overlapping/cluster_1.bin",
      0.01f, 10000, true);
  generateRandomClusterFast<3>(
      "/bigdata/hplsim/production/jungfrau-photoncounter/data_pool/synthetic/"
      "random_clusters_overlapping/cluster_2.bin",
      0.02f, 10000, true);
  generateRandomClusterFast<3>(
      "/bigdata/hplsim/production/jungfrau-photoncounter/data_pool/synthetic/"
      "random_clusters_overlapping/cluster_3.bin",
      0.03f, 10000, true);
  generateRandomClusterFast<3>(
      "/bigdata/hplsim/production/jungfrau-photoncounter/data_pool/synthetic/"
      "random_clusters_overlapping/cluster_4.bin",
      0.04f, 10000, true);
  generateRandomClusterFast<3>(
      "/bigdata/hplsim/production/jungfrau-photoncounter/data_pool/synthetic/"
      "random_clusters_overlapping/cluster_5.bin",
      0.05f, 10000, true);
  generateRandomClusterFast<3>(
      "/bigdata/hplsim/production/jungfrau-photoncounter/data_pool/synthetic/"
      "random_clusters_overlapping/cluster_6.bin",
      0.06f, 10000, true);
  generateRandomClusterFast<3>(
      "/bigdata/hplsim/production/jungfrau-photoncounter/data_pool/synthetic/"
      "random_clusters_overlapping/cluster_7.bin",
      0.07f, 10000, true);
  generateRandomClusterFast<3>(
      "/bigdata/hplsim/production/jungfrau-photoncounter/data_pool/synthetic/"
      "random_clusters_overlapping/cluster_8.bin",
      0.08f, 10000, true);

  // generate random non-overlapping clusters
  generateRandomClusterFast<3>(
      "/bigdata/hplsim/production/jungfrau-photoncounter/data_pool/synthetic/"
      "random_clusters/cluster_4.bin",
      0.04f, 10000, false);*/
  /*generateRandomClusterFast<3>(
      "/bigdata/hplsim/production/jungfrau-photoncounter/data_pool/synthetic/"
      "random_clusters/cluster_8.bin",
      0.08f, 2, false);*/

  // generate test data for other cluster sizes
  /*generateRandomClusterFast<2>(
      "/bigdata/hplsim/production/jungfrau-photoncounter/data_pool/synthetic/"
      "random_clusters_overlapping/cluster2.bin",
      0.0075f, 10000, true);
  generateRandomClusterFast<3>(
      "/bigdata/hplsim/production/jungfrau-photoncounter/data_pool/synthetic/"
      "random_clusters_overlapping/cluster3.bin",
      0.0075f, 10000, true);
  generateRandomClusterFast<7>(
      "/bigdata/hplsim/production/jungfrau-photoncounter/data_pool/synthetic/"
      "random_clusters_overlapping/cluster7.bin",
      0.0075f, 10000, true);
  generateRandomClusterFast<11>(
      "/bigdata/hplsim/production/jungfrau-photoncounter/data_pool/synthetic/"
      "random_clusters_overlapping/cluster11.bin",
      0.0075f, 10000, true);
  */
  
  generateRandomClusterFast<3>(
      "/bigdata/hplsim/production/jungfrau-photoncounter/data_pool/synthetic/nonalign3.bin",
      0.0075f, 10000, true);
  return 0;
}
