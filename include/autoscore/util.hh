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