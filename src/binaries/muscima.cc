#include <assert.h>
#include <experimental/filesystem>
#include <string>
#include <thread>
#include <vector>

#include <autoscore/staff.hh>
#include <autoscore/util.hh>

#define FN_DATASET "../datasets/Handwritten"
#define N_SCORES_PER_W 20
#define N_WRITERS 50

namespace fs = std::experimental::filesystem;

// Threading function
void process_p(std::vector<std::string>::iterator start, const int n_files,
               const int writer, const std::string dist) {

  for (int i = 0; i < n_files; i++, start++) {
    std::cout << *start << std::endl;
    if (!is_image(*start)) {
      continue;
    }
    const std::string output_fn = strip_fn(strip_ext(*start));

    try {
      const cv::Mat img = cv::imread(*start, CV_LOAD_IMAGE_GRAYSCALE);
      const auto model = as::staff::GetStaffModel(img);
      const auto staffs = as::staff::FitStaffModel(model);
      as::staff::SaveToDisk(*start, staffs, model);
      system((std::string("mv ") + output_fn + ".xml " + FN_DATASET + '/' +
              dist + "/w-" + std::to_string(writer) + '/')
                 .c_str());
    } catch (...) {
      std::cerr << "An error occured while processing filename " << *start
                << std::endl;
    }
  }
}

// Program that does staff detection on the MUSCIMA distorted dataset and saves
// the output in a xml format. Line thickness disrotions are not supported
// because they are unreadable. The curvature or rotation distortions are
// supported but for inference only.
int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "Usage : muscima <path-to: /distorsion/> <n_threads>"
              << std::endl;
    return -1;
  }

  int n_threads = 1;
  if (argc == 3) {
    n_threads = atoi(argv[2]);
    assert(n_threads > 0);
  }
  
  // Preparing directories
  // Not checking if they exist because this step is quick whereas
  // DeepScores involves storing all files in a std::map
  // If it already exists, nothing will happen
  const std::vector<std::string> distortions = {
      "kanungo", "ideal",         "interrupted",
      "rotated", "whitespeckles", "typeset-emulation"};
  system((std::string("mkdir ") + FN_DATASET).c_str());
  for (std::string dist : distortions) {
    system((std::string("mkdir ") + FN_DATASET + '/' + dist).c_str());
    for (int i = 1; i <= N_WRITERS; i++) {
      system((std::string("mkdir ") + FN_DATASET + '/' + dist + "/w-" +
              std::to_string(i))
                 .c_str());
    }
  }

  // For every transformation
  const std::string fn = argv[1];
  for (auto &t : fs::directory_iterator(fn)) {
    if (!fs::is_directory(t.path())) {
      continue;
    }
    // Checking if the distortion is one that autoscore can support
    bool is_valid = false;
    std::string distortion;
    for (std::string dist : distortions) {
      if (std::string(t.path()).find(dist) != std::string::npos) {
        is_valid = true;
        distortion = dist;
        break;
      }
    }
    if (!is_valid) {
      continue;
    }

    // For every writer
    for (auto &p : fs::directory_iterator(t.path())) {
      if (!fs::is_directory(p.path())) {
        continue;
      }
      // Finding the writer's number
      std::string writer = p.path();
      const int id_pos = writer.find("w-");
      std::string writer_id;
      for (int i = id_pos + 2; i < id_pos + 2 + 2; i++) {
        writer_id.push_back(writer[i]);
      }
      const int writer_n = atoi(writer_id.c_str());

      // Preparing threading data
      std::vector<std::string> filenames(N_SCORES_PER_W);
      const int files_per_thread = N_SCORES_PER_W / n_threads;
      std::vector<std::thread> threads(n_threads);
      int pos = 0;
      // Storing filenames into a vector
      for (auto &s :
           fs::directory_iterator((std::string(p.path()) + "/image/"))) {
        filenames[pos] = s.path();
        pos++;
      }

      // Splitting the work among threads
      std::cout << std::endl;
      std::cout << p.path() << std::endl;
      auto st = filenames.begin();
      for (int i = 0; i < n_threads; i++, std::advance(st, files_per_thread)) {
        int n_files = files_per_thread;
        if (i == n_threads - 1) {
          n_files = N_SCORES_PER_W - (n_threads - 1) * files_per_thread;
        }
        threads[i] = std::thread(process_p, st, n_files, writer_n, distortion);
      }
      for (auto it = threads.begin(); it != threads.end(); it++) {
        it->join();
      }
    }
  }

  std::cout << "End of DeepScores dataset program" << std::endl;
  return 0;
}