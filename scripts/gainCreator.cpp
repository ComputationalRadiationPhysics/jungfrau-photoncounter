#include <fstream>
#include <iostream>
#include <string>

int main(int argc, char *argv[]) {
  // check argument count
  if (argc != 6) {
    std::cerr << "Usage: gainCreator <width> <height> <gain> <number of gain "
                 "maps> <output path>\n";
    abort();
  }

  // read arguments
  int width = std::atoi(argv[1]);
  int height = std::atoi(argv[2]);
  double gain = std::atof(argv[3]);
  int gainCount = std::atoi(argv[4]);

  // open output file
  std::ofstream outputFile(argv[5], std::ios_base::binary);
  if (!outputFile.is_open()) {
    std::cerr << "Couldn't open output file " << argv[5] << "\n";
    abort();
  }

  // write gain file
  for (int map = 0; map < gainCount; ++map)
    for (int y = 0; y < height; ++y)
      for (int x = 0; x < width; ++x)
        outputFile.write(reinterpret_cast<char *>(&gain), sizeof(gain));

  // close gain file
  outputFile.flush();
  outputFile.close();

  return 0;
}
