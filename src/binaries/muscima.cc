#include <assert.h>
#include <experimental/filesystem>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include <autoscore/staff.hh>
#include <autoscore/util.hh>

#define FN_DATASET "../datasets/Handwritten"

namespace fs = std::experimental::filesystem;

void process_p(std::vector<std::string>::iterator start, const int n_files) {

  for (int i = 0; i < n_files; i++, start++) {
    std::cout << *start << std::endl;
    if (start->find(".png") == std::string::npos &&
        start->find(".jpg") == std::string::npos &&
        start->find(".PNG") == std::string::npos) {
      continue;
    }
    const std::string output_fn = strip_fn(strip_ext(*start));

    try {
      const cv::Mat img = cv::imread(*start, CV_LOAD_IMAGE_GRAYSCALE);
      const auto model = as::staff::GetStaffModel(img);
      const auto staffs = as::staff::FitStaffModel(model);
      as::staff::SaveToDisk(*start, staffs, model);
      system((std::string("mv ") + output_fn + ".xml " + FN_DATASET).c_str());
    } catch (...) {
      std::cerr << "An error occured while processing filename " << *start
                << std::endl;
    }
  }
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "Usage : muscima <relative fn> <nthreads>" << std::endl;
    return -1;
  }

  int n_threads = 1;
  if (argc == 3) {
    n_threads = atoi(argv[2]);
  }
  assert(n_threads > 0);

  std::map<std::string, std::string> processed_images;
  const std::string fn = argv[1];

  if (!fs::exists(FN_DATASET)) {
    system((std::string("mkdir ") + FN_DATASET).c_str());
  } else {
    for (auto &p : fs::directory_iterator(FN_DATASET)) {
      processed_images[strip_fn(strip_ext(p.path()))] = "Exists";
    }
  }

  // For every archive
  for (auto &p : fs::directory_iterator(fn)) {
    if (std::string(p.path()).find(FN_DEEPSCORE) == std::string::npos) {
      continue;
    }
    std::cout << "Working directory : " << p.path() << std::endl;

    if (!fs::exists(std::string(p.path()) + FN_DEEPSCORE_PNG)) {
      std::cout << "No images found in this directory." << std::endl;
      continue;
    }

    const auto start =
        fs::directory_iterator(std::string(p.path()) + FN_DEEPSCORE_PNG);
    const auto finish = end(start);
    const int size = std::distance(start, finish);
    std::cout << size << " files to process among " << n_threads << " threads"
              << std::endl;

    std::vector<std::string> filenames(size);

    const int files_per_thread = size / n_threads;
    std::vector<std::thread> threads(n_threads);

    // Storing filenames into a vector because std::advance does not work on fs
    int pos = 0;
    for (auto &s :
         fs::directory_iterator(std::string(p.path()) + FN_DEEPSCORE_PNG)) {
      if (processed_images.count(strip_fn(strip_ext(s.path())))) {
        std::cout << s.path() << " already processed.\n";
        continue;
      }
      filenames[pos] = s.path();
      pos++;
    }

    std::cout << std::endl;
    std::cout << "Starting dataset processing ..." << std::endl;
    auto st = filenames.begin();
    for (int i = 0; i < n_threads; i++, std::advance(st, files_per_thread)) {
      int n_files = files_per_thread;
      if (i == n_threads - 1) {
        n_files = size - (n_threads - 1) * files_per_thread;
      }
      threads[i] = std::thread(process_p, st, n_files);
    }
    for (auto it = threads.begin(); it != threads.end(); it++) {
      it->join();
    }
  }

  std::cout << "End of Muscima dataset program" << std::endl;
  return 0;
}