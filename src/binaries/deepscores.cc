#include <assert.h>
#include <autoscore/staff.hh>
#include <experimental/filesystem>
#include <map>
#include <string>

#define FN_DATASET "../datasets/Artificial"
#define FN_DEEPSCORE "DeepScores_archive"
#define FN_DEEPSCORE_PNG "/images_png"

namespace fs = std::experimental::filesystem;

std::string strip_fn(const std::string &fn) {
  const int size = fn.size();
  std::string out = fn;
  for (int i = size - 1; i >= 0; i--){
    if (fn[i] == '/'){
      out.clear();
      for (int j = i + 1; j < size; j++){
        out.push_back(fn[j]);
      }
      break;
    }
  }
  return out;
}

std::string strip_ext(const std::string &fn) {
  const int size = fn.size();
  std::string out = fn;
  for (int i = size - 1; i >= 0; i--){
    if (fn[i] == '.'){
      out.clear();
      for (int j = 0; j < i; j++){
        out.push_back(fn[j]);
      }
      break;
    }
  }
  return out;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "Usage : deepscores <relative fn> <nthreads>" << std::endl;
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
      std::cout << strip_fn(strip_ext(p.path())) << std::endl;
      processed_images[strip_fn(strip_ext(p.path()))] = "Exists";
    }
  }

  // For every archive
  for (auto &p : fs::directory_iterator(fn)) {
    std::cout << "Working directory : " << p.path() << std::endl;
    if (std::string(p.path()).find(FN_DEEPSCORE) != std::string::npos) {
      // For every image
      for (auto &i :
           fs::directory_iterator(std::string(p.path()) + FN_DEEPSCORE_PNG)) {
        std::cout << i.path() << std::endl;
        const std::string output_fn =
            strip_fn(strip_ext(fs::absolute(i.path())));

        if (processed_images.count(output_fn)){
          std::cout << "File already exists.\n";
          continue;
        }

        try {
          const cv::Mat img = cv::imread(i.path(), CV_LOAD_IMAGE_GRAYSCALE);
          const auto model = StaffDetect::GetStaffModel(img);
          const auto staffs = StaffDetect::FitStaffModel(model);
          StaffDetect::SaveToDisk(output_fn, staffs, model);
          system(
              (std::string("mv ") + output_fn + ".xml " + FN_DATASET).c_str());
        } catch (...) {
          std::cerr << "An error occured while processing filename " << i.path()
                    << std::endl;
        }
      }
    }
  }

  std::cout << "End of DeepScores dataset program" << std::endl;
  return 0;
}