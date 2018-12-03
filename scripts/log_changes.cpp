#include <fstream>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <string>
#include <memory>

std::unique_ptr<float[]> readData(std::string path) {
  std::unique_ptr<float[]> data(new float[1024 * 512]);

  std::cout << "read file " << path << "\n";
  
  std::ifstream file(path.c_str());
  for(int i = 0; i < 1024 * 512; ++i) {
    file >> data[i];
  }
  
  return data;
}

int main(int argc, char* argv[]) {

  int numberOfFrames = atoi(argv[1]);
  int numberOfPixels = (argc - 2) / 2;
  std::vector<int> x_pos, y_pos;

  for(int i = 0; i < argc - 2; i += 2) {
    x_pos.push_back(atoi(argv[2 + i]));
    y_pos.push_back(atoi(argv[3 + i]));
  }
  
  std::vector<std::vector<float>> pedestals(x_pos.size());
  std::vector<std::vector<float>> detector(x_pos.size());
  
  for(int i = 0; i < numberOfFrames; ++i) {
    std::string path = std::to_string(i) + ".txt";

    std::unique_ptr<float[]> pedestal_data(readData("pedestal_updates_" + path));
    std::unique_ptr<float[]> detector_data(readData("input_" + path));
    
    for(int p = 0; p < numberOfPixels; ++p) {
      pedestals[p].push_back(pedestal_data[y_pos[p] * 1024 + x_pos[p]]);
      detector[p].push_back(detector_data[y_pos[p] * 1024 + x_pos[p]]);
    }
  }

  for(int i = 0; i < numberOfPixels; ++i) {
    std::string path = std::to_string(x_pos[i]) + "_" + std::to_string(y_pos[i]) + ".txt";

    std::ofstream pedestal_file("pedestal_updates_" + path);
    std::ofstream detector_file("input_" + path);
    
    for(unsigned int p = 0; p < pedestals[i].size(); ++p) {
      pedestal_file << std::to_string(pedestals[i][p]) << " ";
      detector_file << std::to_string(detector[i][p]) << " ";
    }

    pedestal_file.flush();
    detector_file.flush();
  }
  
  return EXIT_SUCCESS;
}    
