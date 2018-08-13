#ifndef STAFF_H_
#define STAFF_H_

#include <opencv2/opencv.hpp>
#include <vector>

/**
 * @brief StaffModel that holds the gradient (or orientation) at each column
 * @member std::vector<double> gradient
 * @member int start_col, start_row in the original image
 * @member int staff_height
 * @member int staff_height
 * @member cv::Mat staff_image with removed notes that will serve the fit
 * function
 */
typedef struct staffModel {
  std::vector<double> gradient;
  int start_col, start_row, staff_height, staff_space;
  double rot;
  bool straight;
  cv::Mat staff_image;
} StaffModel;

/**
 * @brief Vector of staffs with first and last line. Combined with a staff
 * model, it can serve pitch inference
 */
typedef std::vector<std::pair<int, int>> Staffs;

namespace as {
namespace staff {

/**
 * @brief Estimates a staff model from a binary image
 * @param cv::Mat src Binary image (CV_8UC1, black on white)
 * @return StaffModel Estimated model
 */
StaffModel GetStaffModel(const cv::Mat &src, const int n_threads = 1);

/**
 * @brief Prints a staff model on a black image
 * @param cv::Mat dst Image on which it will be printed (becomes black)
 */
void PrintStaffModel(cv::Mat &dst, const StaffModel &model);

/**
 * @brief Fits the given model and returns all valid staffs
 * @param StaffModel
 * @return Staffs
 */
Staffs FitStaffModel(const StaffModel &model);

/**
 * @brief Prints all the detected staffs of the model on an image
 * @param cv::Mat dst Image on which it will be printed (will be BGR)
 * @param StaffModel
 * @return Staffs The position of the staffs
 */
void PrintStaffs(cv::Mat &dst, const Staffs &staffs, const StaffModel &model);

/**
 * @brief Removes staffs from an image according to a model and start and end
 * positions
 * @param cv::Mat dst Image with staffs
 * @param Staffs The positions of the staffs
 * @param StaffModel
 */
void RemoveStaffs(cv::Mat &dst, const Staffs &staffs, const StaffModel &model);

/**
 * @brief Realigns the image according to the staff model gradient. The output
 * image is then modeled with a constant gradient. To distort back the image,
 * give the inverse of the model (*-1).
 * @param cv::Mat destination image (grayscale)
 * @param StaffModel the model to work on.
 */
void Realign(cv::Mat &dst, const StaffModel &model);

/**
 * @brief Saves the staff information in a xml format
 * @param std::string fn
 * @param Staffs staffs
 * @param StaffModel model
 */
void SaveToDisk(const std::string &fn, const Staffs &staffs,
                const StaffModel &model);
} // namespace staff

} // namespace as

#endif // STAFF_H_