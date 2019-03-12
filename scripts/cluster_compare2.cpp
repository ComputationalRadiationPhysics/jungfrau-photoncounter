#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <cmath>
#include <vector>
#include <functional>
#include <algorithm>

constexpr std::size_t CLUSTER_SIZE = 3;
constexpr std::size_t MAX_CLUSTER_NUM = 15000;

struct Cluster {
    std::uint64_t frameNumber;
    std::int16_t x;
    std::int16_t y;
    std::int32_t data[CLUSTER_SIZE * CLUSTER_SIZE];
};

struct ClusterArray {
    std::size_t used;
    std::unique_ptr<Cluster[]> clusters;

    explicit ClusterArray(std::size_t maxClusterCount = MAX_CLUSTER_NUM)
        : used(0), clusters(new Cluster[maxClusterCount])
    {
    }
};

struct ClusterFrame {
  explicit ClusterFrame(int32_t frameNumber, std::size_t max_clusters_per_frame = MAX_CLUSTER_NUM) : clusters(max_clusters_per_frame), frameNumber(frameNumber) {}
  ClusterArray clusters;
  int32_t frameNumber;
};

std::vector<ClusterFrame> readTheirClusters(const char* path)
{
  std::vector<ClusterFrame> frames;

    std::cout << "Reading their " << path << " ..." << std::endl;;
    
    std::ifstream clusterFile(path, std::ios::binary);

    std::size_t i = 0;
    std::size_t clusterCount = 0;
    int32_t lastFrameNumber = -1;
    Cluster* clusterPtr = nullptr;



    int flag = 0;


    
    while (clusterFile.good()) {
        // write cluster information
        int32_t frameNumber;
        
        clusterFile.read(reinterpret_cast<char*>(&frameNumber),
                         sizeof(int32_t));
        
        if(lastFrameNumber != frameNumber) {
          if(!frames.empty())
            frames.rbegin()->clusters.used = i - 1;
          i = 0;

          std::cerr << "frame " << frames.size() << std::endl;
          std::cerr << "diff: " << frameNumber - lastFrameNumber << std::endl;
          if(frameNumber - lastFrameNumber > 20 && lastFrameNumber != -1)
            flag = -1; //abort();
          
          
          // insert empty frames
          while(lastFrameNumber != -1 && lastFrameNumber < frameNumber) {
            frames.emplace_back(++lastFrameNumber, 0);
            frames.rbegin()->clusters.used = 0;



            
            if(frames.size() > 45000)
              flag = -1;//abort();

            //std::cerr << "empty frame " << frames.size() << std::endl;
          }



          
            if(frames.size() > 50000)
              flag = -1;//abort();




          
          frames.emplace_back(frameNumber);
          clusterPtr = frames.rbegin()->clusters.clusters.get();
          lastFrameNumber = frameNumber;
        }
   
        clusterPtr[i].frameNumber  = frameNumber & 0xFFFFFFFF;




        
        //if(flag > 5)
        //  abort();
        ++flag;



        
        int x_val, y_val;
        clusterFile.read(reinterpret_cast<char*>(&x_val),
                         sizeof(x_val));
        clusterFile.read(reinterpret_cast<char*>(&y_val),
                         sizeof(y_val));
        clusterPtr[i].x = static_cast<int16_t>(x_val);
        clusterPtr[i].y = static_cast<int16_t>(y_val);



        
        std::cerr << "flag: " << flag << "\t" << clusterPtr[i].frameNumber << "\t" << clusterPtr[i].x << "\t" << clusterPtr[i].y << std::endl;
        if(flag == -1)
          abort();
        
        
        // write cluster
        for (uint8_t y = 0; y < CLUSTER_SIZE; ++y) {
            for (uint8_t x = 0; x < CLUSTER_SIZE; ++x) {
              double clusterValue;
              clusterFile.read(reinterpret_cast<char*>(&clusterValue), sizeof(clusterValue));
              clusterPtr[i].data[x + y * CLUSTER_SIZE] = static_cast<int32_t>(clusterValue);
            }
        }
        ++i;
        ++clusterCount;

        if(i >= MAX_CLUSTER_NUM) {
          std::cerr << "Cluster overflow (over " << MAX_CLUSTER_NUM << " clusters found)!\n";
          exit(EXIT_FAILURE);
        }
    }

    if(0 == i)
      std::cerr << "Warning: No clusters loaded\n";
    
    clusterFile.close();
    
    std::cout << "Read " << clusterCount - 1 << " clusters." << std::endl;
    
    return frames;
}

std::vector<ClusterFrame> readClusters(const char* path)
{
  std::vector<ClusterFrame> frames;

  std::cout << "Reading " << path << " ..." << std::endl;
    
    std::ifstream clusterFile(path, std::ios::binary);

    std::size_t i = 0;
    std::size_t clusterCount = 0;
    int32_t lastFrameNumber = -1;
    Cluster* clusterPtr = nullptr;
    while (clusterFile.good()) {
        // write cluster information
        int32_t frameNumber;
        
        clusterFile.read(reinterpret_cast<char*>(&frameNumber),
                         sizeof(int32_t));
        
        if(lastFrameNumber != frameNumber) {
          if(!frames.empty())
            frames.rbegin()->clusters.used = i - 1;
          i = 0;
          frames.emplace_back(frameNumber);
          clusterPtr = frames.rbegin()->clusters.clusters.get();
          lastFrameNumber = frameNumber;
        }

        clusterPtr[i].frameNumber  = frameNumber & 0xFFFFFFFF;
          
        clusterFile.read(reinterpret_cast<char*>(&clusterPtr[i].x),
                         sizeof(clusterPtr[i].x));
        clusterFile.read(reinterpret_cast<char*>(&clusterPtr[i].y),
                         sizeof(clusterPtr[i].y));

        // write cluster
        for (uint8_t y = 0; y < CLUSTER_SIZE; ++y) {
            for (uint8_t x = 0; x < CLUSTER_SIZE; ++x) {
                clusterFile.read(
                    reinterpret_cast<char*>(
                        &clusterPtr[i].data[x + y * CLUSTER_SIZE]),
                    sizeof(clusterPtr[i].data[x + y * CLUSTER_SIZE]));
            }
        }
        ++i;
        ++clusterCount;

        if(i >= MAX_CLUSTER_NUM) {
          std::cerr << "Cluster overflow (over " << MAX_CLUSTER_NUM << " clusters found)!\n";
          exit(EXIT_FAILURE);
        }
    }

    if(0 == i)
      std::cerr << "Warning: No clusters loaded\n";
    
    clusterFile.close();
    
    std::cout << "Read " << clusterCount - 1 << " clusters." << std::endl;
    
    return frames;
}

bool checkFrameNumbers(std::vector<ClusterFrame> &clusters) {
  std::cout << "Checking frame order ..." << std::endl;

  for(unsigned int i = 1; i < clusters.size(); ++i) {
    if(std::abs(clusters[i-1].frameNumber - clusters[i].frameNumber) > 1)
      return false;
  }
  return true;
}

template<typename T, typename TCmp>
std::tuple<std::size_t, std::size_t> getOffset(T it1, T it2, TCmp cmp) {
  std::size_t offset1 = 0;
  std::size_t offset2 = 0;
  if(cmp(it1->frameNumber, it2->frameNumber)) {
    while(cmp((++it1)->frameNumber, it2->frameNumber))
      ++offset1;
  } else {
    while(cmp((++it2)->frameNumber, it1->frameNumber))
      ++offset2;
  }

  return std::make_tuple(offset1, offset2);
}

std::tuple<std::size_t, std::size_t, std::size_t, std::size_t> selectCommonFrames(const std::vector<ClusterFrame> &v1, const std::vector<ClusterFrame> &v2) {
  const auto start = getOffset(v1.begin(), v2.begin(), std::less_equal<size_t>());
  const auto end = getOffset(v1.begin(), v2.begin(), std::greater_equal<size_t>());
  std::size_t begin1 = std::get<0>(start);
  std::size_t end1 = std::get<0>(end);
  std::size_t begin2 = std::get<1>(start);
  std::size_t end2 = std::get<0>(end);  
  
  return std::make_tuple(begin1, end1, begin2, end2);
}

int main(int argc, char* argv[])
{
    if (argc != 3)
        return EXIT_FAILURE;

    char* detector_path = argv[1];
    char* reference_path = argv[2];

    // read clusters
    std::vector<ClusterFrame> detector = readTheirClusters(detector_path);
    std::vector<ClusterFrame> reference = readClusters(reference_path);

    // check frame numbers
    if(!checkFrameNumbers(detector)) {
      std::cerr << "Error: detector clusters not in order!" << std::endl;
      abort();
    }
    
    if(!checkFrameNumbers(reference)) {
      std::cerr << "Error: detector clusters not in order!" << std::endl;
      abort();
    }

    // extract offset information
    auto offsets = selectCommonFrames(detector, reference);
    size_t detector_begin = std::get<0>(offsets);
    size_t detector_end = detector.size() - std::get<1>(offsets);
    size_t reference_begin = std::get<2>(offsets);
    size_t reference_end = reference.size() - std::get<3>(offsets);

    std::size_t frameCount = reference_end - reference_begin;
    std::cout << "Processing " << frameCount << " common frames!" << std::endl;;
    
    // calculate matches
    std::size_t frameIndex = 0;
    std::vector<std::size_t> exact_matches(frameCount), overlap_matches(frameCount), det_only(frameCount), ref_only(frameCount);
    for(; detector_begin < detector_end && reference_begin < reference_end; ++detector_begin, ++reference_begin) {
      const ClusterArray &referenceClusterArray = reference[reference_begin].clusters;
      const size_t &reference_size = referenceClusterArray.used;
      const Cluster *reference_clusters = referenceClusterArray.clusters.get();
      const ClusterArray &detectorClusterArray = detector[detector_begin].clusters;
      const size_t detector_size = detectorClusterArray.used;
      const Cluster *detector_clusters = detectorClusterArray.clusters.get();

      std::vector<bool> detector_matched(detector_size);
      std::fill(detector_matched.begin(), detector_matched.begin(), false);
      std::vector<bool> reference_matched(reference_size);
      std::fill(reference_matched.begin(), reference_matched.begin(), false);
      
      for(unsigned int det_idx = 0; det_idx < detector_size; ++det_idx) {
        for(unsigned int ref_idx = 0; ref_idx < reference_size; ++ref_idx) {
          const int16_t &ref_x = reference_clusters[ref_idx].x;
          const int16_t &ref_y = reference_clusters[ref_idx].y;
          const int16_t &det_x = detector_clusters[det_idx].x;
          const int16_t &det_y = detector_clusters[det_idx].y;
          if(ref_x == det_x && ref_y == det_y)
            ++exact_matches[frameIndex];
          if(std::pow(ref_x - det_x, 2) + std::pow(ref_y - det_y, 2) < 2) {
            ++overlap_matches[frameIndex];
            reference_matched[ref_idx] = true;
            detector_matched[det_idx] = true;
          }
        }
      }

      det_only[frameIndex] = std::count(detector_matched.begin(), detector_matched.end(), false);
      ref_only[frameIndex] = std::count(reference_matched.begin(), reference_matched.end(), false);
      
      ++frameIndex;
    }

    std::ofstream stats("stats.txt");
    if(!stats.good())
      std::cerr << "Error: Couldn't write stats.txt file!\n";

    for(unsigned int frame = 0; frame < frameCount && stats.good(); ++frame) {
      stats << exact_matches[frame] << " " << overlap_matches[frame] << " " << det_only[frame] << " " << ref_only[frame] << "\n";
    }

    stats.flush();
    stats.close();
    
    return EXIT_SUCCESS;
}
