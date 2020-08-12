#pragma once

#include "jungfrau-photoncounter/Alpakaconfig.hpp"
#include "jungfrau-photoncounter/Config.hpp"

#include <fstream>
#include <memory>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

template <typename Config> class Filecache {
private:
  std::unique_ptr<char, std::function<void(char *)>> buffer;
  char *bufferPointer;
  const std::size_t sizeBytes;
  auto getFileSize(const std::string path) const -> off_t {
    struct stat fileStat;
    stat(path.c_str(), &fileStat);
    return fileStat.st_size;
  }

public:
  Filecache(std::size_t size)
      : buffer(
            [size]() {
              void *p(malloc(size));
              return reinterpret_cast<char *>(p);
            }(),
            free),
        bufferPointer(buffer.get()), sizeBytes(size) {}

  template <typename TData, typename TAlpaka>
  auto loadMaps(const std::string &path, bool header = false)
      -> FramePackage<TData, TAlpaka> {
    // allocate space
    auto fileSize = getFileSize(path);
    std::size_t numFrames = fileSize / sizeof(TData);

    if (fileSize + bufferPointer >= buffer.get() + sizeBytes) {
      if (bufferPointer == 0)
        std::cerr << "Error: Nothing loaded! Is the file path correct?\n";
      else
        std::cerr << "Error: Not enough memory allocated!\n";
      exit(EXIT_FAILURE);
    }

    // load file content
    std::ifstream file;
    file.open(path, std::ios::in | std::ios::binary);
    if (!file.is_open()) {
      std::cerr << "Error: Couldn't open file " << path << "!\n";
      exit(EXIT_FAILURE);
    }

    file.read(bufferPointer, fileSize);
    file.close();

    FramePackage<TData, TAlpaka> maps(numFrames);

    CpuSyncQueue streamBuf = static_cast<CpuSyncQueue>(alpakaGetHost<TAlpaka>());

    TData *dataBuf = reinterpret_cast<TData *>(bufferPointer);

    // copy data into alpaca memory
    alpakaCopy(streamBuf, maps.data,
               alpakaViewPlainPtrHost<TAlpaka, TData>(
                   dataBuf, alpakaGetHost<TAlpaka>(), numFrames),
               numFrames);

    bufferPointer += fileSize;

    return maps;
  }
};
