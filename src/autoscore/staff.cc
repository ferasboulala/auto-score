#include <autoscore/staff.hh>

// c++ std
#include <map>

// c std
#include <assert.h>

// Hyperparameters
#define BINARY_THRESH_VAL 220
// Minimum amount of CC in a column to estimate the gradient
#define MIN_CONNECTED_COMP 5
// Number of neighbouring CC to average the gradient
#define K_NEAREST 5
// Size of the sliding window in number of lines to find staffs
#define KERNEL_SIZE 5
// 1/ratio of the model must be polled
#define MIN_POLL_RATIO_STRAIGHT 4
#define MIN_POLL_RATIO_CURVED 4
// Ratio of the max amount of lines per staff to suspect the presence of a staff
#define LINE_PER_STAFF_RATIO_STRAIGHT 0.5
#define LINE_PER_STAFF_RATIO_CURVED 0.5
// Minimum amount of detected HoughLines to consider it straight
#define MIN_HOUGH_LINES 10

// Hough
#define THETA_RES 2
#define N_BINS 20

// Useful constants
#define LINES_PER_STAFF 5
#define RAD2DEG 180 / CV_PI
#define DEG2RAD 1 / (RAD2DEG)
#define EPSILON 1e-7

namespace {

inline bool isGray(const cv::Mat &src) {
  unsigned char depth = src.type() & CV_MAT_DEPTH_MASK;
  if (depth != CV_8U || (src.channels() != 1))
    return false;
  return true;
}

inline void boundingBox(const cv::Mat &src, cv::Mat &dst) {
  cv::Mat points;
  cv::findNonZero(src, points);
  cv::Rect bbox = cv::boundingRect(points);
  dst = dst(bbox);
}

void drawModel(cv::Mat &dst, const StaffModel &model, const int pos,
               const cv::Scalar color) {
  assert(!isGray(dst));
  double y = pos;
  for (auto it = model.gradient.begin(); it != model.gradient.end(); it++) {
    y += *it;
    const int x = it - model.gradient.begin() + model.start_col;
    if ((y > dst.rows) || (y < 0))
      continue;
    dst.at<cv::Vec3b>((int)round(y), x)[0] = color[0];
    dst.at<cv::Vec3b>((int)round(y), x)[1] = color[1];
    dst.at<cv::Vec3b>((int)round(y), x)[2] = color[2];
  }
}

void rotateImage(cv::Mat &dst, const double rot_theta) {
  cv::Point2f center((dst.cols - 1) / 2.0, (dst.rows - 1) / 2.0);
  cv::Mat rot = cv::getRotationMatrix2D(center, rot_theta, 1.0);
  cv::Rect bbox =
      cv::RotatedRect(cv::Point2f(), dst.size(), rot_theta).boundingRect();
  rot.at<double>(0, 2) += bbox.width / 2.0 - dst.cols / 2.0;
  rot.at<double>(1, 2) += bbox.height / 2.0 - dst.rows / 2.0;
  cv::warpAffine(dst, dst, rot, bbox.size());
}

bool blackOnWhite(const cv::Mat &src) {
  int black = 0, white = 0;
  for (int i = 0; i < src.rows / 2; i++) {
    for (int j = 0; j < src.cols / 2; j++) {
      if (src.at<char>(i, j))
        white++;
      else
        black++;
    }
  }
  if (black > white)
    return false;
  return true;
}

} // namespace

StaffModel StaffDetect::GetStaffModel(const cv::Mat &src) {
  assert(isGray(src));

  cv::Mat img;
  src.copyTo(img);
  if (blackOnWhite(img))
    cv::threshold(img, img, BINARY_THRESH_VAL, 255, CV_THRESH_BINARY_INV);

  StaffModel model;

  // If HoughTransform yields a lot of 90 degrees lines, the model will not be
  // infered
  std::vector<cv::Vec2f> lines;
  cv::HoughLines(img, lines, 1, CV_PI / (180 * THETA_RES), img.cols / 2);

  // Storing them into a histogram
  std::vector<std::vector<cv::Vec2f>> theta_count(N_BINS,
                                                  std::vector<cv::Vec2f>());
  for (auto line : lines) {
    const double theta = line[1] * RAD2DEG;
    const int idx = (int)theta / (int)(CV_PI * RAD2DEG / N_BINS);
    theta_count[idx].push_back(line);
  }

  // Getting the most populated bin from the histogram
  int max_count = 0, max_index = 0;
  for (int i = 0; i < theta_count.size(); i++) {
    if (theta_count[i].size() > max_count) {
      max_index = i;
      max_count = theta_count[i].size();
    }
  }

  // Getting the average of that bin
  double avg_theta = 0;
  int diag = sqrt(pow(img.cols, 2) + pow(img.rows, 2));
  for (auto line : theta_count[max_index]) {
    const double theta = line[1];
    avg_theta += theta;
  }
  avg_theta /= (theta_count[max_index].size() + EPSILON);

  // If 70% of the lines are in it and there are at least more than 10 lines,
  // use a constant model (no gradient) (most of the time)
  model.rot = CV_PI / 2;
  model.straight = false;
  if ((double)theta_count[max_index].size() / lines.size() >= 0.7 &&
      lines.size() > MIN_HOUGH_LINES) {
    rotateImage(img, RAD2DEG * (avg_theta - CV_PI / 2));
    model.straight = true;
    model.rot = avg_theta;
  }

  cv::Mat points;
  cv::findNonZero(img, points);
  cv::Rect bbox = cv::boundingRect(points);
  img = img(bbox);
  model.start_col = bbox.x;
  model.start_row = bbox.y;

  if (model.straight) {
    std::vector<double> gradient(img.cols, 0.0);
    model.gradient = gradient;
  }

  // Getting an estimate of staff_height and staff_space
  // map< count, poll >
  std::map<int, int> white_run_length; // text
  std::map<int, int> black_run_length; // spaces
  for (int x = 0; x < img.cols; x++) {
    int val = img.at<char>(0, x);
    int count = 1;
    for (int y = 1; y < img.rows; y++) {
      if (!((val == 0) == (img.at<char>(y, x) == 0))) {
        if (!val)
          black_run_length[count]++;
        else
          white_run_length[count]++;
        val = img.at<char>(y, x);
        count = 1;
      } else
        count++;
    }
  }

  // staff_height and staff_space are assigned to the most polled runs
  int staff_height, staff_space;
  int max_polled = 0;
  for (auto it : white_run_length) {
    if (max_polled < it.second) {
      max_polled = it.second;
      staff_height = it.first;
    }
  }
  max_polled = 0;
  for (auto it : black_run_length) {
    if (max_polled < it.second) {
      max_polled = it.second;
      staff_space = it.first;
    }
  }

  // assert(staff_height < staff_space);
  model.staff_height = staff_height;
  model.staff_space = staff_space;

  // Removing symbols based on estimated staff_height
  cv::Mat staff_image = img;
  // If you do img.copyTo(staff_image), you get fucked up errors
  const int T = staff_height +
                1; // std::min(2 * staff_height, staff_height + staff_space);
  for (int x = 0; x < staff_image.cols; x++) {
    int val = staff_image.at<char>(0, x);
    int count = 1;
    for (int y = 1; y < staff_image.rows; y++) {
      if ((val == 0) != (staff_image.at<char>(y, x) == 0)) {
        if (val != 0) {
          if (count > T) {
            for (int k = y - 1; k >= y - count - 1; k--)
              staff_image.at<char>(k, x) = 0;
          }
        }
        count = 1;
        val = staff_image.at<char>(y, x);
      } else
        count++;
    }
  }

  model.staff_image = staff_image; // shared_ptr
  if (model.straight)
    return model;

  // Getting all connected components of each column
  struct ConnectedComponent {
    int n, x, y;
  };
  std::vector<std::vector<struct ConnectedComponent>> components(
      staff_image.cols);
  for (int x = 0; x < staff_image.cols; x++) {
    int count = 1;
    int val = staff_image.at<char>(0, x);
    for (int y = 1; y < staff_image.rows; y++) {
      if ((val == 0) != (staff_image.at<char>(y, x) == 0) ||
          y == staff_image.rows - 1) { // optional condition
        if (val != 0) {
          struct ConnectedComponent cc;
          cc.x = x;
          cc.y = y - 1;
          cc.n = count;
          components[x].push_back(cc);
        }
        val = staff_image.at<char>(y, x);
        count = 1;
      } else
        count++;
    }
  }

  // Computing the orientation at each column
  std::vector<double> orientations(staff_image.cols, staff_image.rows);
  // For every column
  for (int x = 0; x < staff_image.cols; x++) {
    double global_orientation = 0;
    int global_count = 0;
    if (components[x].size() < MIN_CONNECTED_COMP)
      continue;
    // For every connected component in that column
    for (auto cc : components[x]) {
      double local_orientation = 0;
      int local_count = 0;
      // For every K nearest component
      for (int k = 1; k <= K_NEAREST; k++) {
        const int next_idx = k + x;
        if (next_idx >= staff_image.cols)
          break;
        double row_dist = staff_image.rows;
        for (auto next_cc = components[next_idx].begin();
             next_cc != components[next_idx].end(); next_cc++) {
          // If we are getting closer
          if (abs(row_dist) > abs(next_cc->y - cc.y)) {
            row_dist =
                (next_cc->y - (int)(next_cc->n / 2)) - (cc.y - (int)(cc.n / 2));
          } else
            break;
        }
        if (abs(row_dist) <= k) { // Not in paper
          local_orientation += row_dist / k;
          local_count++;
        }
      }
      if (local_count) {
        global_orientation += local_orientation / local_count;
        global_count++;
      }
    }
    if (global_count) {
      orientations[x] = global_orientation / global_count;
    }
  }

  // Interpolating with empty columns
  double prev_orientation = staff_image.rows,
         next_orientation = staff_image.rows;
  for (int i = 0; i < orientations.size(); i++) {
    if (orientations[i] != staff_image.rows) {
      prev_orientation = orientations[i];
      continue;
    }
    int current = i;
    while (i < orientations.size()) {
      if (orientations[i] != staff_image.rows) {
        next_orientation = orientations[i];
        break;
      }
      i++;
    }
    // If one orientation is undefined, copy their interpolation (slope = 0)
    if ((prev_orientation == staff_image.rows) &&
        (next_orientation != staff_image.rows)) {
      prev_orientation = next_orientation;
    } else if ((prev_orientation != staff_image.rows) &&
               (next_orientation == staff_image.rows)) {
      next_orientation = prev_orientation;
    }
    const double delta_slope =
        (next_orientation - prev_orientation) / (i - current);
    for (int j = current; j < i; j++) {
      orientations[j] = prev_orientation + (i - j) * delta_slope;
    }
  }

  model.gradient = orientations;

  return model;
}

void StaffDetect::PrintStaffModel(cv::Mat &dst, const StaffModel &model) {
  const double rotation = RAD2DEG * (model.rot - CV_PI / 2);
  assert(isGray(dst));
  rotateImage(dst, -rotation);
  boundingBox(dst, dst);
  dst =
      cv::Mat(cv::Size(model.gradient.size(), model.gradient.size()), CV_8UC3);
  drawModel(dst, model, dst.rows / 2, cv::Scalar(255, 0, 0));
}

Staffs StaffDetect::FitStaffModel(const StaffModel &model) {
  cv::Mat img = model.staff_image;
  const bool straight = model.straight;

  // Polling each staff line and keep only the ones
  const int n_rows = img.rows;
  const int n_cols = model.gradient.size();
  std::vector<bool> staff_lines;
  for (int y = 0; y < n_rows; y++) {
    int poll = 0;
    double estimated_y = y;
    for (auto it = model.gradient.begin(); it != model.gradient.end(); it++) {
      estimated_y += *it;
      const int x = it - model.gradient.begin() + model.start_col;
      const int rounded_y = round(estimated_y);
      // Boudary check
      if (estimated_y > n_rows || estimated_y < 0)
        continue;
      // Model fits
      else if (img.at<char>(rounded_y, x)) {
        poll++;
      }
      // Check if the model fits with a staff_height padding around it
      else if (!model.straight) {
        for (int i = 1; i <= model.staff_height; i++) {
          if (estimated_y + i < n_rows || estimated_y - i >= 0) {
            if (img.at<char>(rounded_y + i, x) ||
                img.at<char>(rounded_y - i, x)) {
              poll++;
              break;
            }
          }
        }
      }
    }
    bool is_line = poll >= n_cols / MIN_POLL_RATIO_STRAIGHT;
    if (!straight) {
      is_line = poll >= n_cols / MIN_POLL_RATIO_CURVED;
    }
    staff_lines.push_back(is_line);
  }

  // Checking how many duplicates per staff line on average (+1)
  double avg_duplicate = 0;
  int n_lines = 0;
  for (int i = 0; i < staff_lines.size(); i++) {
    if (!staff_lines[i])
      continue;
    int dup = 0;
    const int cur_pos = i;
    for (int j = i; j < staff_lines.size(); j++) {
      const bool is_line = staff_lines[j];
      const int next_pos = j;
      if (next_pos - cur_pos <= model.staff_height && is_line) {
        dup++;
      } else {
        i = j;
        break;
      }
    }
    avg_duplicate += dup;
    n_lines++;
  }
  avg_duplicate /= (n_lines + EPSILON);
  // Add an assertion for avg_duplicate >= model.staff_height

  // Convolving a 1-D kernel
  // The local maximas are zones where there might be a staff
  Staffs staffs;
  const int kernel = KERNEL_SIZE * model.staff_height +
                     (KERNEL_SIZE - 1) * model.staff_space +
                     round(avg_duplicate);
  // higher --> harder on scarce staffs
  // lower --> staffs will start anywhere
  int min_lines_per_staff = 0;
  // higher --> will end anywhere
  // lower --> Harder with lyrics
  const int staff_size = (LINES_PER_STAFF)*model.staff_height +
                         (LINES_PER_STAFF - 1) * model.staff_space;
  for (int i = 0; i < img.rows; i++) {
    int count = 0;
    for (int j = 0; j + i < img.rows && j < kernel; j++) {
      const int idx = i + j;
      if (staff_lines[idx])
        count++;
    }
    if (count > min_lines_per_staff)
      min_lines_per_staff = count;
  }

  // Higher --> Some staffs will not be detected
  // Lower --> Staffs will start too early (on hight pitch notes for instance)
  if (!straight)
    min_lines_per_staff *= LINE_PER_STAFF_RATIO_CURVED;
  else
    min_lines_per_staff *= LINE_PER_STAFF_RATIO_STRAIGHT;

  // Hysteresis
  for (int i = 0; i < img.rows; i++) {
    int count = 0;
    for (int j = 0; j + i < img.rows && j < kernel; j++) {
      if (staff_lines[i + j])
        count++;
    }

    if (count >= min_lines_per_staff) {
      int flag = 0;
      int next_count = min_lines_per_staff;
      std::vector<int> maxes;
      while (i < img.rows && flag <= 2 * model.staff_space) {
        next_count = 0;
        for (int j = 0; j + i < img.rows && j < kernel; j++) {
          const int idx = j + i;
          if (staff_lines[idx])
            next_count++;
        }
        if (next_count == count) {
          maxes.push_back(i);
        } else if (next_count > count) {
          maxes.clear();
          maxes.push_back(i);
          count = next_count;
        } else if (next_count < min_lines_per_staff) {
          flag++;
        }
        i++;
      }
      double start = 0;
      for (int idx : maxes) {
        start += idx;
      }
      start = round(start / maxes.size());
      if (!staff_lines[start]) {
        int k = 1;
        while (k <= model.staff_space + model.staff_height &&
               start + k < staff_lines.size() && start - k >= 0) {
          if (staff_lines[start - k]) {
            const int l = k;
            while (staff_lines[start - k] && start - k >= 0) {
              k++;
            }
            start -= (k + l) / 2;
            break;
          } else if (staff_lines[start + k]) {
            const int l = k;
            while (staff_lines[start + k] && start + k < staff_lines.size()) {
              k++;
            }
            start += (k + l) / 2;
            break;
          }
          k++;
        }
      }
      const int finish = start + staff_size;
      i = finish + model.staff_space;
      staffs.push_back(std::pair<int, int>(start, finish));
    }
  }

  return staffs;
}

void StaffDetect::PrintStaffs(cv::Mat &dst, const Staffs &staffs,
                              const StaffModel model) {
  assert(isGray(dst));
  const double rotation = RAD2DEG * (model.rot - CV_PI / 2);
  rotateImage(dst, rotation);
  cv::cvtColor(dst, dst, CV_GRAY2BGR);
  dst *= 0.5;
  for (auto s : staffs) {
    const double staff_interval = s.second - s.first;
    for (int i = 1; i < LINES_PER_STAFF - 1; i++) {
      const int line_pos = round(staff_interval / (LINES_PER_STAFF - 1) * i) +
                           s.first + model.start_row;
      drawModel(dst, model, line_pos, cv::Scalar(255, 0, 0));
    }
    int line_pos = s.first + model.start_row;
    drawModel(dst, model, line_pos, cv::Scalar(0, 255, 0));
    line_pos =
        round(staff_interval / (LINES_PER_STAFF - 1) * (LINES_PER_STAFF - 1)) +
        s.first + model.start_row;
    drawModel(dst, model, line_pos, cv::Scalar(0, 0, 255));
  }
}

// TODO : Add relevant assertions !!!
// TODO : Add RemoveStaff()