#include <autoscore/staff.hh>

// c++ std
#include <map>
#include <thread>

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
#define MIN_POLL_RATIO_STRAIGHT 5
#define MIN_POLL_RATIO_CURVED 10
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

inline bool is_gray(const cv::Mat &src) {
  unsigned char depth = src.type() & CV_MAT_DEPTH_MASK;
  if (depth != CV_8U || (src.channels() != 1))
    return false;
  return true;
}

inline void bounding_box(const cv::Mat &src, cv::Mat &dst) {
  cv::Mat points;
  cv::findNonZero(src, points);
  cv::Rect bbox = cv::boundingRect(points);
  dst = dst(bbox);
}

void draw_model(cv::Mat &dst, const StaffModel &model, const int pos,
                const cv::Scalar color) {
  assert(!is_gray(dst));
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

void rotate_image(cv::Mat &dst, const double rot_theta) {
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

void estimate_rotation(cv::Mat &img, StaffModel &model) {
  // If HoughTransform yields a lot of 90 degrees lines, the model will not be
  // infered. The straight line will be the chosen model
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
    rotate_image(img, RAD2DEG * (avg_theta - CV_PI / 2));
    model.straight = true;
    model.rot = avg_theta;
  }
  cv::Mat points;
  cv::findNonZero(img, points);
  cv::Rect bbox = cv::boundingRect(points);
  img = img(bbox);
  model.start_col = bbox.x;
  model.start_row = bbox.y;
}

struct RunLengthData {
  cv::Mat img;
  std::vector<int> *whites, *blacks;
  int start, finish;
};

void run_length_p(struct RunLengthData &data) {
  const cv::Mat img = data.img;
  for (int x = data.start; x < data.finish; x++) {
    int val = img.at<char>(0, x);
    int count = 1;
    for (int y = 1; y < img.rows; y++) {
      if (!((val == 0) == (img.at<char>(y, x) == 0))) {
        if (!val)
          (*data.blacks)[count]++;
        else
          (*data.whites)[count]++;
        val = img.at<char>(y, x);
        count = 1;
      } else
        count++;
    }
  }
}

void run_length(const cv::Mat &img, int &staff_height, int &staff_space,
                const int n_threads) {
  std::vector<std::vector<int>> white_run_length(n_threads,
                                                 std::vector<int>(img.rows, 0));
  std::vector<std::vector<int>> black_run_length(n_threads,
                                                 std::vector<int>(img.rows, 0));
  std::vector<std::thread> threads(n_threads);
  std::vector<struct RunLengthData> run_length_data(n_threads);

  const int cols_per_thread = img.cols / n_threads;
  for (int i = 0; i < n_threads; i++) {
    const int start = i * cols_per_thread;
    int end = (i + 1) * cols_per_thread;
    if (i == n_threads - 1)
      end = img.cols;
    run_length_data[i].img = img;
    run_length_data[i].start = start;
    run_length_data[i].finish = end;
    run_length_data[i].whites = &white_run_length[i];
    run_length_data[i].blacks = &black_run_length[i];
    threads[i] = std::thread(run_length_p, std::ref(run_length_data[i]));
  }
  for (auto it = threads.begin(); it != threads.end(); it++) {
    it->join();
  }

  // staff_height and staff_space are assigned to the most polled runs
  std::vector<int> white_poll(img.rows, 0), black_poll(img.rows, 0);
  for (int i = 0; i < img.rows; i++) {
    for (int j = 0; j < n_threads; j++) {
      white_poll[i] += white_run_length[j][i];
      black_poll[i] += black_run_length[j][i];
    }
  }
  int max_polled = 0;
  for (auto it = white_poll.begin(); it != white_poll.end(); it++) {
    if (max_polled < *it) {
      max_polled = *it;
      staff_height = it - white_poll.begin();
    }
  }
  max_polled = 0;
  for (auto it = black_poll.begin(); it != black_poll.end(); it++) {
    if (max_polled < *it) {
      max_polled = *it;
      staff_space = it - black_poll.begin();
    }
  }
}

void remove_glyphs(cv::Mat &staff_image, const int staff_height,
                   const int staff_space) {
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
}

void estimate_gradient(StaffModel &model) {
  // Getting all connected components of each column
  cv::Mat staff_image = model.staff_image;
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
  model.gradient = orientations;
}

void interpolate_model(StaffModel &model) {
  auto &orientations = model.gradient;
  cv::Mat staff_image = model.staff_image;
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
}

std::vector<int> poll_lines(const StaffModel &model) {
  cv::Mat img = model.staff_image;
  const bool straight = model.straight;

  // Polling each staff line and keep only the ones
  const int n_rows = img.rows;
  const int n_cols = model.gradient.size();
  std::vector<int> staff_lines;
  for (int y = 0; y < n_rows; y++) {
    int poll = 0;
    double estimated_y = y;
    for (auto it = model.gradient.begin(); it != model.gradient.end(); it++) {
      estimated_y += *it;
      const int x = it - model.gradient.begin() + model.start_col;
      const int rounded_y = round(estimated_y);
      // Boundary check
      if (estimated_y > n_rows || estimated_y < 0)
        continue;
      // Model fits
      else if (img.at<char>(rounded_y, x)) {
        poll++;
      }
      // Check if the model fits with a staff_height padding around it
      else if (!model.straight) {
        for (int i = 1; i <= model.staff_height; i++) {
          if (rounded_y + i < n_rows || rounded_y - i >= 0) {
            if (img.at<char>(rounded_y + i, x) ||
                img.at<char>(rounded_y - i, x)) {
              poll++;
              break;
            }
          }
        }
      }
    }
    int is_line = poll * (int)(poll >= n_cols / MIN_POLL_RATIO_STRAIGHT);
    if (!straight) {
      is_line = poll * (int)(poll >= n_cols / MIN_POLL_RATIO_CURVED);
    }
    staff_lines.push_back(is_line);
  }
  return staff_lines;
}

} // namespace

StaffModel StaffDetect::GetStaffModel(const cv::Mat &src, const int n_threads) {
  assert(is_gray(src));
  assert(n_threads > 0);

  cv::Mat img;
  src.copyTo(img);
  if (blackOnWhite(img))
    cv::threshold(img, img, BINARY_THRESH_VAL, 255, CV_THRESH_BINARY_INV);
  else {
    //cv::threshold(img, img, 255 - BINARY_THRESH_VAL, 255, CV_THRESH_BINARY);
  }

  StaffModel model;

  // Checking whether it is straight or not
  estimate_rotation(img, model);
  if (model.straight) {
    std::vector<double> gradient(img.cols, 0.0);
    model.gradient = gradient;
  }

  // Getting an estimate of staff_height and staff_space
  int staff_height, staff_space;
  run_length(img, staff_height, staff_space, n_threads);

  model.staff_height = staff_height;
  model.staff_space = staff_space;

  // Removing symbols based on estimated staff_height
  cv::Mat staff_image = img;
  remove_glyphs(staff_image, staff_height, staff_space);
  model.staff_image = staff_image; // shared_ptr
  if (model.straight)
    return model;

  estimate_gradient(model);
  interpolate_model(model);

  return model;
}

void StaffDetect::PrintStaffModel(cv::Mat &dst, const StaffModel &model) {
  const double rotation = RAD2DEG * (model.rot - CV_PI / 2);
  rotate_image(dst, -rotation);
  bounding_box(dst, dst);
  if (is_gray(dst)) {
    dst = cv::Mat(cv::Size(model.gradient.size(), model.gradient.size()),
                  CV_8UC3);
  }
  draw_model(dst, model, dst.rows / 2, cv::Scalar(255, 0, 0));
}

Staffs StaffDetect::FitStaffModel(const StaffModel &model) {
  cv::Mat img = model.staff_image;
  const bool straight = model.straight;

  // Polling each staff line and keeping the ones that are polled enough
  std::vector<int> staff_lines = poll_lines(model);

  // Convolving a 1-D kernel
  // The local maximas are zones where there might be a staff
  Staffs staffs;
  const int kernel = KERNEL_SIZE * model.staff_height +
                     (KERNEL_SIZE - 1) * model.staff_space + model.staff_space;
  // higher --> harder on scarce staffs
  // lower --> staffs will start anywhere
  int min_lines_per_staff = 0;
  // higher --> will end anywhere
  // lower --> Harder with lyrics
  const int staff_size = (LINES_PER_STAFF - 0.5) * model.staff_height +
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
      while (i < img.rows && flag <= model.staff_space) {
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
        } else {
          flag++;
        }
        i++;
      }
      double start = 0;
      for (int idx : maxes) {
        start += idx;
      }
      start = round(start / maxes.size());
      int cur_poll = staff_lines[start];
      int offset = 0;
      for (int k = 0;
           k < model.staff_space - model.staff_height && start - k >= 0 && start + k < img.rows;
           k++) {
        if (cur_poll < staff_lines[start - k]){
          offset = -k;
          cur_poll = staff_lines[start - k];
        }
        if (cur_poll < staff_lines[start + k]){
          offset = k;
          cur_poll = staff_lines[start + k];
        }
      }
      start += offset;
      const int finish = start + staff_size;
      i = finish + model.staff_space;
      staffs.push_back(std::pair<int, int>(start, finish));
    }
  }
  return staffs;
}

void StaffDetect::PrintStaffs(cv::Mat &dst, const Staffs &staffs,
                              const StaffModel model) {
  const double rotation = RAD2DEG * (model.rot - CV_PI / 2);
  rotate_image(dst, rotation);
  if (is_gray(dst)) {
    cv::cvtColor(dst, dst, CV_GRAY2BGR);
  }
  dst *= 0.5;
  for (auto s : staffs) {
    const double staff_interval = s.second - s.first;
    for (int i = 1; i < LINES_PER_STAFF - 1; i++) {
      const int line_pos = round(staff_interval / (LINES_PER_STAFF - 1) * i) +
                           s.first + model.start_row;
      draw_model(dst, model, line_pos, cv::Scalar(255, 0, 0));
    }
    int line_pos = s.first + model.start_row;
    draw_model(dst, model, line_pos, cv::Scalar(0, 255, 0));
    line_pos =
        round(staff_interval / (LINES_PER_STAFF - 1) * (LINES_PER_STAFF - 1)) +
        s.first + model.start_row;
    draw_model(dst, model, line_pos, cv::Scalar(0, 0, 255));
  }
}