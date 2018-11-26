#ifndef _UTIL_HH
#define _UTIL_HH

#include <string>

/**
 * @brief Strips the path of a filename
 * @param std::string fn : absolute or relative filename
 * @return std::string name of the file
*/
std::string strip_fn(const std::string &fn);

/**
 * @brief Strips the extension of a file
 * @param std::string fn : path
 * @return std::string path without the extension
*/
std::string strip_ext(const std::string &fn);

/**
 * @brief Checks if the filename is an image supported by OpenCV
 * @param std::string fn : The filename
 * @return bool
*/
inline bool is_image(const std::string &fn){
  return !(fn.find(".png") == std::string::npos &&
        fn.find(".jpg") == std::string::npos &&
        fn.find(".PNG") == std::string::npos);
}

#endif // _UTIL_HH