#include "Filecache.hpp"

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>


auto Filecache::getFileSize(const std::string path) const -> off_t
{
    struct stat fileStat;
    stat(path.c_str(), &fileStat);
    return fileStat.st_size;
}

Filecache::Filecache(std::size_t size)
    : buffer([size]() {
          void* p(malloc(size));
          return reinterpret_cast<char*>(p);
      }()),
      bufferPointer(buffer.get()), sizeBytes(size)
{
}
