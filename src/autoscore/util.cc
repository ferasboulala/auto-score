#include <autoscore/util.hh>

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