#pragma once

#include <cstdlib>
#include <fstream>
#include <optional.hpp>
#include <string>
#include <vector>

#include "jungfrau-photoncounter/AlpakaHelper.hpp"

#include "jungfrau-photoncounter/Debug.hpp"

// struct for all reference paths
struct ResultCheck {
  std::string photonPath;
  std::string energyPath;
  std::string maxValuesPath;
  std::string sumPath;
  std::string clusterPath;
};

// checks an arbitrary frame package with a binary file
template <typename T>
bool checkResult(tl::optional<T> result, std::string referencePath) {
  // check if data is present
  if (!result || referencePath == "" || referencePath == "_")
    return true;

  // load photon reference data
  std::ifstream reference_file(referencePath, std::ios::binary);
  if (!reference_file.is_open()) {
    std::cerr << "Couldn't open reference file " << referencePath << "!\n";
    return false;
  }

  // seek beginning of file
  reference_file.seekg(0);

  // determine read size
  using TData =
      typename alpaka::traits::ElemType<decltype(result->data)>::type;
  std::size_t dataTypeSize = sizeof(TData);
  std::size_t extent = alpakaGetExtent<0>(result->data);

  std::size_t dataSize = extent * dataTypeSize;
  std::size_t mapSize = sizeof(TData::data) / sizeof(TData::data[0]);

  // read reference data
  T reference(result->numFrames);
  reference_file.read(reinterpret_cast<char *>(alpakaNativePtr(reference.data)),
                      dataSize);
  std::size_t readBytes = reference_file.gcount();
  if (reference_file.bad() || reference_file.fail()) {
    std::cerr << "Read failed!\n";
    return false;
  }

  // check if the correct number of bytes was read
  char dummyByte;
  if (readBytes < dataSize || !reference_file.read(&dummyByte, 1).eof()) {
    std::cerr << "Number of read bytes does not match (" << readBytes << " vs. "
              << dataSize << "; eof: " << reference_file.eof() << ")!\n";
    return false;
  }

  // compare data
  bool exactly_identical = true;
  bool very_close = true;
  std::size_t closeCount = 0;
  std::size_t oneCount = 0;
  for (std::size_t frameNumber = 0; frameNumber < extent; ++frameNumber) {
    // check header
    if (alpakaNativePtr(result->data)[frameNumber].header.frameNumber !=
            alpakaNativePtr(reference.data)[frameNumber].header.frameNumber &&
        alpakaNativePtr(result->data)[frameNumber].header.bunchId !=
            alpakaNativePtr(reference.data)[frameNumber].header.bunchId) {
      std::cout << "Header mismatch in frame " << frameNumber << "\n";
      return false;
    }

    for (std::size_t index = 0; index < mapSize; ++index) {
      // check if data is exactly identical
      if (alpakaNativePtr(result->data)[frameNumber].data[index] !=
          alpakaNativePtr(reference.data)[frameNumber].data[index]) {
        exactly_identical = false;
        ++closeCount;
      }

      // check if data is very close
      if (std::abs(alpakaNativePtr(result->data)[frameNumber].data[index] -
                   alpakaNativePtr(reference.data)[frameNumber].data[index]) >
          0.001) {
        very_close = false;
        ++oneCount;
      }

      // check if data is of by at most one
      if (std::abs(alpakaNativePtr(result->data)[frameNumber].data[index] -
                   alpakaNativePtr(reference.data)[frameNumber].data[index]) >
          1.0) {

        save_image<JungfrauConfig>("res_" + std::to_string(frameNumber),
                                   alpakaNativePtr(result->data), frameNumber);
        save_image<JungfrauConfig>("ref_" + std::to_string(frameNumber),
                                   alpakaNativePtr(reference.data),
                                   frameNumber);

        std::cout << "Result mismatch in frame " << frameNumber << " on index "
                  << index << " ("
                  << alpakaNativePtr(result->data)[frameNumber].data[index]
                  << " vs. "
                  << alpakaNativePtr(reference.data)[frameNumber].data[index]
                  << ") \n";

        // return false;
      }
    }
  }

  // close file and return
  reference_file.close();

  if (exactly_identical)
    std::cout << "Data is completely identical. \n";
  else if (very_close)
    std::cout << "Data is very close (" << closeCount << " not identical). \n";
  else
    std::cout << "Data is at most off by one (" << closeCount
              << " not identical, " << oneCount << " off by at most one). \n";

  return true;
}

// checks an arbitrary frame package with a binary file
template <typename T>
bool checkResultRaw(tl::optional<T> result, std::string referencePath) {
  // check if data is present
  if (!result || referencePath == "" || referencePath == "_")
    return true;

  // load photon reference data
  std::ifstream reference_file(referencePath, std::ios::binary);
  if (!reference_file.is_open()) {
    std::cerr << "Couldn't open reference file " << referencePath << "!\n";
    return false;
  }

  // seek beginning of file
  reference_file.seekg(0);

  // determine read size
  using TData =
      typename alpaka::traits::ElemType<decltype(result->data)>::type;
  std::size_t dataTypeSize = sizeof(TData);
  std::size_t extent = alpakaGetExtent<0>(result->data);

  std::size_t dataSize = extent * dataTypeSize;

  // read reference data
  T reference(result->numFrames);
  reference_file.read(reinterpret_cast<char *>(alpakaNativePtr(reference.data)),
                      dataSize);
  std::size_t readBytes = reference_file.gcount();

  if (reference_file.bad() || reference_file.fail()) {
    std::cerr << "Read failed!\n";
    return false;
  }

  // check if the correct number of bytes was read
  char dummyByte;
  if (readBytes < dataSize || !reference_file.read(&dummyByte, 1).eof()) {
    std::cerr << "Number of read bytes does not match (" << readBytes << " vs. "
              << dataSize << "; eof: " << reference_file.eof() << ")!\n";
    return false;
  }

  // compare data
  bool exactly_identical = true;
  bool very_close = true;
  for (std::size_t frameNumber = 0; frameNumber < extent; ++frameNumber) {
    // check if data is exactly identical
    if (alpakaNativePtr(result->data)[frameNumber] !=
        alpakaNativePtr(reference.data)[frameNumber]) {
      exactly_identical = false;
    }

    // check if data is very close
    if (std::abs(alpakaNativePtr(result->data)[frameNumber] -
                 alpakaNativePtr(reference.data)[frameNumber]) > 0.001) {
      very_close = false;
    }

    // check if data is of by at most one
    if (std::abs(alpakaNativePtr(result->data)[frameNumber] -
                 alpakaNativePtr(reference.data)[frameNumber]) > 1.0) {

      std::cerr << "Result mismatch in frame " << frameNumber << " ("
                << alpakaNativePtr(result->data)[frameNumber] << " vs. "
                << alpakaNativePtr(reference.data)[frameNumber] << ") \n";

      return false;
    }
  }

  // close file and return
  reference_file.close();

  if (exactly_identical)
    std::cout << "Data is completely identical. \n";
  else if (very_close)
    std::cout << "Data is very close. \n";
  else
    std::cout << "Data is at most off by one. \n";

  return true;
}

// holds clusters from one frame
template <class TClusterArray> struct ClusterFrame {
  explicit ClusterFrame(int32_t frameNumber, std::size_t maxClusters)
      : frameNumber(frameNumber), clusters(maxClusters) {}
  TClusterArray clusters;
  int32_t frameNumber;
};

// read clusters from file
template <class TClusterArray, unsigned CLUSTER_SIZE>
std::vector<ClusterFrame<TClusterArray>> readClusters(const char *path,
                                                      std::size_t maxClusters) {
  using Cluster = typename alpaka::traits::ElemType<decltype(
      TClusterArray::clusters)>::type;

  // allocate space
  std::vector<ClusterFrame<TClusterArray>> frames;

  std::cout << "Reading " << path << " ...\n";

  // open file
  std::ifstream clusterFile(path, std::ios::binary);

  // initialize counters
  std::size_t i = 0;
  std::size_t clusterCount = 0;
  int32_t lastFrameNumber = -1;
  Cluster *clusterPtr = nullptr;
  while (clusterFile.good()) {
    // write cluster information
    int32_t frameNumber;

    // read frame number
    clusterFile.read(reinterpret_cast<char *>(&frameNumber), sizeof(int32_t));

    // allocate new frame if necessary
    if (lastFrameNumber != frameNumber) {
      if (!frames.empty())
        frames.rbegin()->clusters.used = i;
      i = 0;
      frames.emplace_back(frameNumber, maxClusters);
      clusterPtr = alpakaNativePtr(frames.rbegin()->clusters.clusters);
      lastFrameNumber = frameNumber;
    }

    // set correct frame number in cluster
    clusterPtr[i].frameNumber = frameNumber & 0xFFFFFFFF;

    // read the position of the cluster
    clusterFile.read(reinterpret_cast<char *>(&clusterPtr[i].x),
                     sizeof(clusterPtr[i].x));
    clusterFile.read(reinterpret_cast<char *>(&clusterPtr[i].y),
                     sizeof(clusterPtr[i].y));

    // read the actual cluster
    for (uint8_t y = 0; y < CLUSTER_SIZE; ++y) {
      for (uint8_t x = 0; x < CLUSTER_SIZE; ++x) {
        clusterFile.read(
            reinterpret_cast<char *>(&clusterPtr[i].data[x + y * CLUSTER_SIZE]),
            sizeof(clusterPtr[i].data[x + y * CLUSTER_SIZE]));
      }
    }

    // increment counters
    ++i;
    ++clusterCount;
  }

  if (0 == i)
    std::cerr << "Warning: No clusters loaded\n";

  clusterFile.close();

  std::cout << "Read " << clusterCount - 1 << " clusters.\n";

  return frames;
}

// get start and end offset
template <typename T, typename TCmp>
std::tuple<std::size_t, std::size_t> getOffset(T it1, T it2, TCmp cmp) {
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

// check common frames
template <class ClusterFrame>
bool checkCommonFrames(const std::vector<ClusterFrame> &v1,
                       const std::vector<ClusterFrame> &v2) {

  // check if the length of both vectors differ
  if (v1.size() != v2.size())
    return false;

  // check if all frame numbers match
  for (std::size_t i = 0; i < v1.size(); ++i) {
    if (v1[i].frameNumber != v2[i].frameNumber) {
      std::cerr << "Error: Frame numbers mismatch: " << v1[i].frameNumber
                << " vs. " << v2[i].frameNumber << "\n";
      return false;
    }
  }

  return true;
}

// convert clusters from memory
template <class TClusterArray, unsigned CLUSTER_SIZE>
std::vector<ClusterFrame<TClusterArray>>
convertClusters(TClusterArray &clusters, std::size_t maxClusters) {
  using Cluster = typename alpaka::traits::ElemType<decltype(
      TClusterArray::clusters)>::type;

  // get cluster pointer
  Cluster *inputClusterPtr = alpaka::getPtrNative(clusters.clusters);
  Cluster *clusterPtr = nullptr;

  // allocate space
  std::vector<ClusterFrame<TClusterArray>> frames;

  // initialize counters
  std::size_t i = 0;
  std::size_t clusterCount = 0;
  int32_t lastFrameNumber = -1;

  // iterate over clusters
  for (std::uint64_t c = 0; c < clusters.used; ++c) {
    // get frame number
    int32_t frameNumber = inputClusterPtr[c].frameNumber;

    // allocate new frame if necessary
    if (lastFrameNumber != frameNumber) {
      if (!frames.empty())
        frames.rbegin()->clusters.used = i;

      i = 0;
      frames.emplace_back(frameNumber, maxClusters);
      clusterPtr = alpakaNativePtr(frames.rbegin()->clusters.clusters);
      lastFrameNumber = frameNumber;
    }

    // set frame number to current cluster
    clusterPtr[i].frameNumber = frameNumber & 0xFFFFFFFF;
    clusterPtr[i].x = inputClusterPtr[c].x;
    clusterPtr[i].y = inputClusterPtr[c].y;

    // copy actual cluster
    for (unsigned y = 0; y < CLUSTER_SIZE; ++y)
      for (unsigned x = 0; x < CLUSTER_SIZE; ++x)
        clusterPtr[i].data[x + y * CLUSTER_SIZE] =
            inputClusterPtr[c].data[x + y * CLUSTER_SIZE];

    // increment counters
    ++clusterCount;
    ++i;
  }

  std::cout << "Read " << clusterCount << " clusters.\n";

  return frames;
}

// compares a cluster array with stored reference data
template <class TConfig, class TAlpaka>
bool checkClusters(typename TConfig::template ClusterArray<TAlpaka> *result,
                   std::string referencePath, std::size_t maxClusters = 50000) {

  // check if data is present
  if (!result || referencePath == "" || referencePath == "_")
    return true;

  // load and convert clusters
  std::vector<ClusterFrame<typename TConfig::template ClusterArray<TAlpaka>>>
      clusters_result(
          convertClusters<typename TConfig::template ClusterArray<TAlpaka>,
                          TConfig::CLUSTER_SIZE>(*result, maxClusters));

  std::vector<ClusterFrame<typename TConfig::template ClusterArray<TAlpaka>>>
      clusters_reference(
          readClusters<typename TConfig::template ClusterArray<TAlpaka>,
                       TConfig::CLUSTER_SIZE>(referencePath.c_str(),
                                              maxClusters));

  // extract offset information
  if (!checkCommonFrames(clusters_result, clusters_reference)) {
    std::cerr << "Error: Results differ in length!\n";
    return false;
  }

  // calculate matches
  std::size_t frameCount = clusters_reference.size();
  std::size_t frameIndex = 0;
  bool exactly_equal = true;
  std::vector<std::size_t> exact_matches(frameCount),
      overlap_matches(frameCount), res_only(frameCount), ref_only(frameCount);

  for (std::size_t frame = 0; frame < clusters_reference.size(); ++frame) {
    // get reference cluster array for the current frame
    const typename TConfig::template ClusterArray<TAlpaka>
        &referenceClusterArray = clusters_reference[frame].clusters;
    const size_t &reference_size = referenceClusterArray.used;
    const typename TConfig::Cluster *reference_clusters =
        alpakaNativePtr(referenceClusterArray.clusters);

    // get result cluster array for the current frame
    const typename TConfig::template ClusterArray<TAlpaka> &resultClusterArray =
        clusters_result[frame].clusters;
    const size_t result_size = resultClusterArray.used;
    const typename TConfig::Cluster *result_clusters =
        alpakaNativePtr(resultClusterArray.clusters);

    // create arrays to store which clusters have been matched
    std::vector<bool> result_matched(result_size);
    std::fill(result_matched.begin(), result_matched.begin(), false);
    std::vector<bool> reference_matched(reference_size);
    std::fill(reference_matched.begin(), reference_matched.begin(), false);

    // actually match the clusters
    for (unsigned int res_idx = 0; res_idx < result_size; ++res_idx) {
      for (unsigned int ref_idx = 0; ref_idx < reference_size; ++ref_idx) {
        // get the position of the reference and the result cluster center
        const int16_t &ref_x = reference_clusters[ref_idx].x;
        const int16_t &ref_y = reference_clusters[ref_idx].y;
        const int16_t &res_x = result_clusters[res_idx].x;
        const int16_t &res_y = result_clusters[res_idx].y;

        // check for exact match
        if (ref_x == res_x && ref_y == res_y) {
          ++exact_matches[frameIndex];

          // check if the cluster values are close
          for (unsigned cly = 0; cly < TConfig::CLUSTER_SIZE; ++cly) {
            for (unsigned clx = 0; clx < TConfig::CLUSTER_SIZE; ++clx) {

              // check for exact equality
              if (reference_clusters[ref_idx]
                      .data[cly * TConfig::CLUSTER_SIZE + clx] !=
                  result_clusters[res_idx]
                      .data[cly * TConfig::CLUSTER_SIZE + clx])
                exactly_equal = false;

              // check for inequality
              if (std::abs(reference_clusters[ref_idx]
                               .data[cly * TConfig::CLUSTER_SIZE + clx] -
                           result_clusters[res_idx]
                               .data[cly * TConfig::CLUSTER_SIZE + clx]) >
                  0.001f) {
                std::cerr << "Error: Value mismatch on frame " << frame
                          << " on position y=" << cly << ", x=" << clx << ": "
                          << result_clusters[ref_idx]
                                 .data[cly * TConfig::CLUSTER_SIZE + clx]
                          << " vs. "
                          << reference_clusters[res_idx]
                                 .data[cly * TConfig::CLUSTER_SIZE + clx]
                          << "\n";

                return false;
              }
            }
          }
        }

        // check for overlap in general
        if (std::pow(ref_x - res_x, 2) + std::pow(ref_y - res_y, 2) < 2) {
          ++overlap_matches[frameIndex];
          reference_matched[ref_idx] = true;
          result_matched[res_idx] = true;
        }
      }
    }

    // determine which clusters are exclusive to either the reference or the
    // result
    res_only[frameIndex] =
        std::count(result_matched.begin(), result_matched.end(), false);
    ref_only[frameIndex] =
        std::count(reference_matched.begin(), reference_matched.end(), false);

    // abort if either reference or result exclusive clusters exist
    if (res_only[frameIndex] > 0 || ref_only[frameIndex] > 0) {
      std::cerr << "Error: Exclusive clusters found!\n";
      return false;
    }

    ++frameIndex;
  }

  // give feedback about equality
  if (exactly_equal)
    std::cout << "All matches were exactly equal!\n";
  else
    std::cout << "Some matches were slightly off!\n";

  return true;
}
