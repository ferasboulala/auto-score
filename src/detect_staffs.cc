#include <autoscore/staff.hh>
#include <chrono>
#include <experimental/filesystem>
#include <string>

#define WINDOW_HEIGHT 1440

namespace fs = std::experimental::filesystem;

void process_image(const std::string &fn, const int n_threads) {
  cv::Mat img = cv::imread(fn, CV_LOAD_IMAGE_GRAYSCALE);

  auto start = std::chrono::high_resolution_clock::now();

  auto model = StaffDetect::GetStaffModel(img, n_threads);
  cv::Mat no_staffs;
  img.copyTo(no_staffs);
  auto staffs = StaffDetect::FitStaffModel(no_staffs, model, true);

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float> duration = end - start;
  std::chrono::duration_cast<std::chrono::milliseconds>(duration);
  std::cout << "Took : " << duration.count() << std::endl;

  StaffDetect::PrintStaffs(img, staffs, model);

  const std::string identified = "Staffs";
  const std::string removed = "Staffs";
  const double ratio = (double)img.cols / img.rows;
  cv::namedWindow(identified, CV_WINDOW_NORMAL);
  cv::resizeWindow(identified, WINDOW_HEIGHT, ratio * WINDOW_HEIGHT);
  cv::imshow(identified, img);
  cv::waitKey(0);
  cv::namedWindow(removed, CV_WINDOW_NORMAL);
  cv::resizeWindow(removed, WINDOW_HEIGHT, ratio * WINDOW_HEIGHT);
  cv::imshow(removed, no_staffs);
  cv::waitKey(0);
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "Usage : detect_staffs <input> (optional: <nthreads>)"
              << std::endl;
    return -1;
  }
  int n_threads = 1;
  if (argc == 3) {
    n_threads = atoi(argv[2]);
  }

  std::string fn = argv[1];
  auto not_found = std::string::npos;
  if (fn.find(".png") != not_found || fn.find(".jpg") != not_found) {
    process_image(fn, n_threads);
    return 0;
  }

  for (auto &p : fs::directory_iterator(fn)) {
    std::cout << p.path() << std::endl;
    process_image(p.path(), n_threads);
  }

  return 0;
}