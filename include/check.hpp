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
    std::cerr << "Couldn't open referencefile " << referencePath << "!\n";
    return false;
  }

  // seek beginning of file
  reference_file.seekg(0);

  // determine read size
  using TData =
      typename alpaka::elem::traits::ElemType<decltype(result->data)>::type;
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
  for (std::size_t frameNumber = 0; frameNumber < extent; ++frameNumber) {
    for (std::size_t index = 0; index < mapSize; ++index) {
      // check if data is exactly identical
      if (alpakaNativePtr(result->data)[frameNumber].data[index] !=
          alpakaNativePtr(reference.data)[frameNumber].data[index]) {
        exactly_identical = false;
      }

      // check if data is very close
      if (std::abs(alpakaNativePtr(result->data)[frameNumber].data[index] -
                   alpakaNativePtr(reference.data)[frameNumber].data[index]) >
          0.001) {
        very_close = false;
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

        std::cerr << "Result mismatch in frame " << frameNumber << " on index "
                  << index << " ("
                  << alpakaNativePtr(result->data)[frameNumber].data[index]
                  << " vs. "
                  << alpakaNativePtr(reference.data)[frameNumber].data[index]
                  << ") \n";

        return false;
      }
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

// checks an arbitrary frame package with a binary file
template <typename T>
bool checkResultRaw(tl::optional<T> result, std::string referencePath) {
  // check if data is present
  if (!result || referencePath == "" || referencePath == "_")
    return true;

  // load photon reference data
  std::ifstream reference_file(referencePath, std::ios::binary);
  if (!reference_file.is_open()) {
    std::cerr << "Couldn't open referencefile " << referencePath << "!\n";
    return false;
  }

  // seek beginning of file
  reference_file.seekg(0);

  // determine read size
  using TData =
      typename alpaka::elem::traits::ElemType<decltype(result->data)>::type;
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
      save_single_map<JungfrauConfig>(
          "res_" + std::to_string(frameNumber),
          &alpakaNativePtr(result->data)[frameNumber]);
      save_single_map<JungfrauConfig>(
          "ref_" + std::to_string(frameNumber),
          &alpakaNativePtr(reference.data)[frameNumber]);

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

// compares a cluster array with stored reference data
template <typename T> bool checkClusters(T *result, std::string referencePath) {
  // check if data is present
  if (!result || referencePath == "" || referencePath == "_")
    return true;

  std::cerr << "TODO: implement a good cluster comparing algorithm!\n";

  return false;
}
