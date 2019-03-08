#include <cstdlib>
#include <fstream>
#include <sstream>
#include <iostream>

constexpr uint8_t CLUSTER_SIZE = 3;

struct Cluster {
    int32_t frameNumber;
    int16_t coord_x;
    int16_t coord_y;
    int32_t data[CLUSTER_SIZE * CLUSTER_SIZE];
} cluster;

int main(int argc, char* argv[])
{
    if (argc != 3) {
        std::cerr << "Invalid arguments!\nUsage: converter <input> <output>"
                  << std::endl;
        return EXIT_FAILURE;
    }

    std::ifstream in(argv[1], std::ios::binary);
    std::ofstream out(argv[2]);
    std::stringstream str;

    if (!in.is_open()) {
        std::cerr << "Couldn't open file " << argv[1] << "!" << std::endl;
        return EXIT_FAILURE;
    }

    if (!out.is_open()) {
        std::cerr << "Couldn't open file " << argv[2] << "!" << std::endl;
        return EXIT_FAILURE;
    }

    uint64_t frameCount = 0;
    
    while(!in.eof()) {
        // read cluster information
        in.read(reinterpret_cast<char*>(&cluster), sizeof(Cluster));

        // write cluster information
        str << cluster.frameNumber << "\n\t" << cluster.coord_x << "\n\t"
            << cluster.coord_y << "\n";

        // write cluster
        for (uint8_t y = 0; y < CLUSTER_SIZE; ++y) {
          str << "\t";
            for (uint8_t x = 0; x < CLUSTER_SIZE; ++x) {
                str << cluster.data[x + y * CLUSTER_SIZE] << " ";
            }

            str << "\n";
        }
        ++frameCount;
    }

    out << frameCount << "\n";
    out << str.str();

    out.flush();
    in.close();
    out.close();
    
    return EXIT_SUCCESS;
}
