#include "Filecache.hpp"
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

off_t Filecache::getFileSize(const std::string path) const
{
    struct stat fileStat;
    stat(path.c_str(), &fileStat);
    return fileStat.st_size;
}

Filecache::Filecache(std::size_t sizeBytes)
    : buffer([sizeBytes]() {
          void* p;
          cudaMallocHost(&p, sizeBytes);
          return reinterpret_cast<char*>(p);
      }(), cudaFreeHost),
      bufferPointer(buffer.get()), sizeBytes(sizeBytes)
{
}
