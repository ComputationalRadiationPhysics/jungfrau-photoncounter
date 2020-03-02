#include <fstream>
#include <iostream>
#include <vector>

constexpr std::size_t CLUSTER_SIZE = 3;

struct Cluster {
    std::uint32_t frameNumber;
    std::int16_t x;
    std::int16_t y;
    double data[CLUSTER_SIZE * CLUSTER_SIZE];
};

std::vector<Cluster> readClusters(const char* path)
{
  std::vector<Cluster> clusters;

  std::cout << "Reading " << path << " ...\n";

  std::ifstream clusterFile(path, std::ios::binary);
  
  while(!clusterFile.eof()) {
    Cluster c;
    clusterFile.read(reinterpret_cast<char*>(&c), sizeof(c));
    clusters.emplace_back(c);
    
    if(!clusterFile){
      std::cerr << "File reading went wrong!\n";
      std::cerr << "fail: " << clusterFile.fail() << " bad: " << clusterFile.bad() << " eof: " << clusterFile.eof() << "\n";
      //abort();

      clusters.pop_back();
      break;
    }
  }

  return clusters;
}

void saveClustersASCII(const std::string& path, const std::vector<Cluster>& clusters)
{
  std::ofstream clusterFile(path.c_str());
  
  clusterFile << clusters.size() << "\n";

  for(const Cluster& c : clusters) {
    clusterFile << c.frameNumber << "\n";
    clusterFile << "\t" << c.x << " " << c.y << "\n";
    
    for(unsigned int y = 0; y < CLUSTER_SIZE; ++y) {
      clusterFile << "\t";
      for(unsigned int x = 0; x < CLUSTER_SIZE; ++x)
        clusterFile << c.data[x + y * CLUSTER_SIZE] << " ";
      
      clusterFile << "\n";
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

    const std::vector<Cluster> clusters = readClusters(input);
    saveClustersASCII(output, clusters);

    return EXIT_SUCCESS;
}
