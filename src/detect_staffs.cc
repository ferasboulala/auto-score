#include <autoscore/staff.hh>
#include <experimental/filesystem>
#include <string>

#define WINDOW_HEIGHT 1440

namespace fs = std::experimental::filesystem;

void process_image(const std::string &fn, const bool save = false) {
  cv::Mat img = cv::imread(fn, CV_LOAD_IMAGE_GRAYSCALE);
  auto model = StaffDetect::GetStaffModel(img);
  auto staffs = StaffDetect::FitStaffModel(model);
  StaffDetect::PrintStaffs(img, staffs, model);

  const std::string window_name = "Staffs";
  const double ratio = (double)img.cols / img.rows;
  cv::namedWindow(window_name, CV_WINDOW_NORMAL);
  cv::resizeWindow(window_name, WINDOW_HEIGHT, ratio * WINDOW_HEIGHT);
  cv::imshow(window_name, img);
  cv::waitKey(0);

  if (!save)
    return;

  cv::imwrite("anotated_" + fn, img);
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "Usage : detect_staffs <input>" << std::endl;
    return -1;
  }

  std::string fn = argv[1];
  auto not_found = std::string::npos;
  if (fn.find(".png") != not_found || fn.find(".jpg") != not_found) {
    process_image(fn, true);
    return 0;
  }

  for (auto &p : fs::directory_iterator(fn)) {
    std::cout << p.path() << std::endl;
    process_image(p.path());
  }

  return 0;
}