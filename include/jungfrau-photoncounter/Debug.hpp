#pragma once

#include <alpaka/alpaka.hpp>
#include <chrono>
#include <fstream>
#include <iostream>

#include "Config.hpp"

template <typename TConfig, typename TAlpaka>
void saveClusters(std::string path,
                  typename TConfig::template ClusterArray<TAlpaka> &clusters) {
  //#if (NDEBUG)
  std::ofstream clusterFile;
  clusterFile.open(path);
  clusterFile << clusters.used << "\n";
  typename TConfig::Cluster *clusterPtr =
      alpaka::getPtrNative(clusters.clusters);

  DEBUG("writing", clusters.used, "clusters to", path);

  for (uint64_t i = 0; i < clusters.used; ++i) {
    // write cluster information
    clusterFile << static_cast<uint32_t>(clusterPtr[i].frameNumber & 0xFFFFFFFF)
                << "\n\t" << clusterPtr[i].x << " " << clusterPtr[i].y << "\n";

    // write cluster
    for (uint8_t y = 0; y < TConfig::CLUSTER_SIZE; ++y) {
      clusterFile << "\t";
      for (uint8_t x = 0; x < TConfig::CLUSTER_SIZE; ++x) {
        clusterFile << clusterPtr[i].data[x + y * TConfig::CLUSTER_SIZE] << " ";
      }

      clusterFile << "\n";
    }
  }

  clusterFile.close();
  //#endif
}

template <typename TConfig, typename TBuffer>
void save_image(std::string path, TBuffer *data, std::size_t frame_number) {
  //#if (NDEBUG)
  std::ofstream img;
  img.open(path + ".txt");
  for (std::size_t j = 0; j < TConfig::DIMY; j++) {
    for (std::size_t k = 0; k < TConfig::DIMX; k++) {
      double h = double(data[frame_number].data[(j * TConfig::DIMX) + k]);
      img << h << " ";
    }
    img << "\n";
  }
  img.close();
  //#endif
}

template <typename TConfig, typename TBuffer>
void save_single_map(std::string path, TBuffer *data) {
  //#if (NDEBUG)
  std::ofstream img;
  img.open(path + ".txt");
  for (std::size_t j = 0; j < TConfig::DIMY; j++) {
    for (std::size_t k = 0; k < TConfig::DIMX; k++) {
      double h = double(data[(j * TConfig::DIMX) + k]);
      img << h << " ";
    }
    img << "\n";
  }
  img.close();
  //#endif
}

template <typename TConfig, typename TAlpaka>
void saveClusterArray(
    std::string path,
    std::vector<typename TConfig::template ClusterArray<TAlpaka>> &clusters) {
  //#if (NDEBUG)
  std::ofstream clusterFile;
  clusterFile.open(path);

  uint64_t numClusters = 0;
  for (const auto &clusterArray : clusters)
    numClusters += clusterArray.used;

  clusterFile << numClusters << "\n";

  DEBUG("writing", numClusters, "clusters to", path);

  for (auto &clusterArray : clusters) {
    typename TConfig::Cluster *clusterPtr =
        alpaka::getPtrNative(clusterArray.clusters);

    for (uint64_t i = 0; i < clusterArray.used; ++i) {
      // write cluster information
      clusterFile << static_cast<int32_t>(clusterPtr[i].frameNumber &
                                          0xFFFFFFFF)
                  << "\n\t" << clusterPtr[i].x << "\n\t" << clusterPtr[i].y
                  << "\n";

      // write cluster
      for (uint8_t y = 0; y < TConfig::CLUSTER_SIZE; ++y) {
        clusterFile << "\t";
        for (uint8_t x = 0; x < TConfig::CLUSTER_SIZE; ++x) {
          clusterFile << clusterPtr[i].data[x + y * TConfig::CLUSTER_SIZE]
                      << " ";
        }

        clusterFile << "\n";
      }
    }
  }

  clusterFile.close();
  //#endif
}

template <typename TConfig, typename TAlpaka>
void saveClustersBin(
    std::string path,
    typename TConfig::template ClusterArray<TAlpaka> &clusters) {
//#if (NDEBUG)
  std::ofstream clusterFile(path.c_str(), std::ios::binary);
  typename TConfig::Cluster *clusterPtr =
      alpaka::getPtrNative(clusters.clusters);

  DEBUG("writing", clusters.used, "clusters to", path);

  for (uint64_t i = 0; i < clusters.used; ++i) {
    // write cluster information
    int32_t frameNumber = clusterPtr[i].frameNumber & 0xFFFFFFFF;
    clusterFile.write(reinterpret_cast<char *>(&frameNumber), sizeof(int32_t));
    clusterFile.write(reinterpret_cast<char *>(&clusterPtr[i].x),
                      sizeof(clusterPtr[i].x));
    clusterFile.write(reinterpret_cast<char *>(&clusterPtr[i].y),
                      sizeof(clusterPtr[i].y));

    // write cluster
    for (uint8_t y = 0; y < TConfig::CLUSTER_SIZE; ++y) {
      for (uint8_t x = 0; x < TConfig::CLUSTER_SIZE; ++x) {
        clusterFile.write(
            reinterpret_cast<char *>(
                &clusterPtr[i].data[x + y * TConfig::CLUSTER_SIZE]),
            sizeof(clusterPtr[i].data[x + y * TConfig::CLUSTER_SIZE]));
      }
    }
  }

  clusterFile.flush();
  clusterFile.close();
  //#endif
}

struct Point {
  uint16_t x, y;
};

template <typename TConfig, typename Accelerator> class PixelTracker {
private:
  std::vector<Point> positions;
  std::vector<std::vector<double>> input, pedestal[TConfig::PEDEMAPS],
      stddev[TConfig::PEDEMAPS];

public:
  PixelTracker(int argc, char *argv[]) {
    int numberOfPixels = (argc - 1) / 2;
    DEBUG("adding", numberOfPixels, "pixels");
    for (int i = 0; i < numberOfPixels; ++i) {
      DEBUG(atoi(argv[2 * i + 1]), ":", atoi(argv[2 * i + 2]));
      addPixel(atoi(argv[2 * i + 1]), atoi(argv[2 * i + 2]));
    }
  }

  PixelTracker(std::vector<Point> positions = std::vector<Point>())
      : positions(positions) {}

  void addPixel(Point position) {
    positions.push_back(position);

    if (position.x >= TConfig::DIMX || position.y >= TConfig::DIMY) {
      std::cerr << "Pixel out of range!" << std::endl;
      abort();
    }

    input.resize(input.size() + 1);
    for (std::size_t p = 0; p < TConfig::PEDEMAPS; ++p) {
      pedestal[p].resize(input.size() + 1);
      stddev[p].resize(input.size() + 1);
    }
  }

  void addPixel(uint16_t x, uint16_t y) { addPixel({x, y}); }

  void push_back(
      typename TConfig::template FramePackage<typename TConfig::InitPedestalMap,
                                              Accelerator>
          init_pedestals,
      typename TConfig::template FramePackage<typename TConfig::PedestalMap,
                                              Accelerator>
          raw_pedestals,
      typename TConfig::template FramePackage<typename TConfig::DetectorData,
                                              Accelerator>
          raw_input,
      size_t offset) {
    for (int i = 0; i < input.size(); ++i) {
      input[i].push_back(
          alpaka::getPtrNative(raw_input.data)[offset]
              .data[positions[i].y * TConfig::DIMX + positions[i].x]);
      DEBUG("input",
            alpaka::getPtrNative(raw_input.data)[offset]
                .data[positions[i].y * TConfig::DIMX + positions[i].x]);
      for (std::size_t p = 0; p < TConfig::PEDEMAPS; ++p) {
        pedestal[p][i].push_back(alpaka::getPtrNative(
            raw_pedestals
                .data)[p][positions[i].y * TConfig::DIMX + positions[i].x]);
        DEBUG("pede[", p, "][", i, "]",
              alpaka::getPtrNative(
                  raw_pedestals.data)[p][positions[i].y * TConfig::DIMX +
                                         positions[i].x]);
        stddev[p][i].push_back(
            alpaka::getPtrNative(
                init_pedestals
                    .data)[p][positions[i].y * TConfig::DIMX + positions[i].x]
                .stddev);
        DEBUG("stddev[", p, "][", i, "]",
              alpaka::getPtrNative(
                  init_pedestals
                      .data)[p][positions[i].y * TConfig::DIMX + positions[i].x]
                  .stddev);
      }
    }
  }

  void save() {
    DEBUG("saving", input.size(), "pixels");

    for (int i = 0; i < input.size(); ++i) {
      DEBUG("saving pixel", std::to_string(positions[i].x) + ":" +
                                std::to_string(positions[i].y));

      std::ofstream input_file("input_" + std::to_string(positions[i].x) + "_" +
                               std::to_string(positions[i].y) + ".txt");

      std::ofstream pedestal_file[TConfig::PEDEMAPS];
      std::ofstream stddev_file[TConfig::PEDEMAPS];

      for (std::size_t p = 0; p < TConfig::PEDEMAPS; ++p) {
        pedestal_file[p].open("pedestal_" + std::to_string(p) + "_" +
                              std::to_string(positions[i].x) + "_" +
                              std::to_string(positions[i].y) + ".txt");

        stddev_file[p].open("stddev_" + std::to_string(p) + "_" +
                            std::to_string(positions[i].x) + "_" +
                            std::to_string(positions[i].y) + ".txt");
      }

      for (unsigned int j = 0; j < input[i].size(); ++j) {
        input_file << input[i][j] << " ";
        for (std::size_t p = 0; p < TConfig::PEDEMAPS; ++p) {
          pedestal_file[p] << pedestal[p][i][j] << " ";
          stddev_file[p] << stddev[p][i][j] << " ";
        }
      }

      input_file.flush();
      input_file.close();

      for (std::size_t p = 0; p < TConfig::PEDEMAPS; ++p) {
        pedestal_file[p].flush();
        pedestal_file[p].close();
        stddev_file[p].flush();
        stddev_file[p].close();
      }
    }
  }
};
