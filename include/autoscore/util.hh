#ifndef _UTIL_HH
#define _UTIL_HH

#include <string>

/**
 * \fn std::string strip_fn(const std::string &fn)
 * \brief Strips the path of a filename
 * \param fn Absolute or relative filename
 * \return Name of the file
*/
std::string strip_fn(const std::string &fn);

/**
 * \fn std::string strip_ext(const std::string &fn)
 * \brief Strips the extension of a file
 * \param fn Path to the file
 * \return Path without the extension
*/
std::string strip_ext(const std::string &fn);

/**
 * \fn inline bool is_image(const std::string &fn)
 * \brief Checks if the filename is an image supported by OpenCV
 * \param fn The filename
 * \return A boolean
*/
inline bool is_image(const std::string &fn){
  return !(fn.find(".png") == std::string::npos &&
        fn.find(".jpg") == std::string::npos &&
        fn.find(".PNG") == std::string::npos);
}

#endif // _UTIL_HH