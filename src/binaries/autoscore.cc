#include <autoscore/staff.hh>
#include <autoscore/util.hh>
#include <opencv2/opencv.hpp>

#include <experimental/filesystem>
#include <string>

namespace fs = std::experimental::filesystem;

void process_image(const std::string &fn, const int n_threads) {
  cv::Mat img = cv::imread(fn, CV_LOAD_IMAGE_GRAYSCALE);
  auto model = as::staff::GetStaffModel(img, n_threads);
  auto staffs = as::staff::FitStaffModel(model);
  as::staff::SaveToDisk("output", staffs, model);
  as::staff::PrintStaffs(img, staffs, model);
  cv::imwrite("../pictures/staff_" + strip_fn(fn), img);
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "Usage : autoscore <input> (optional: <nthreads>)"
              << std::endl;
    return -1;
  }
  int n_threads = 1;
  if (argc == 3) {
    n_threads = atoi(argv[2]);
  }

  std::string fn = argv[1];
  auto not_found = std::string::npos;
  if (is_image(fn)) {
    process_image(fn, n_threads);
    return 0;
  }

  for (auto &p : fs::directory_iterator(fn)) {
    std::cout << p.path() << std::endl;
    process_image(p.path(), n_threads);
  }

  return 0;
}