cmake_minimum_required(VERSION 2.8)

project(auto-score)

find_package(OpenCV REQUIRED)

include_directories(include)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_STANDARD_11)
set(ALL_LIBRARIES)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")

list(APPEND ALL_LIBRARIES ${OpenCV_LIBS})
list(APPEND ALL_LIBRARIES -lstdc++fs)
list(APPEND ALL_LIBRARIES -pthread)

macro(link_whole_archive lib)
  if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    list(APPEND ALL_LIBRARIES -Wl,-force_load,$<TARGET_FILE_NAME:${lib}> ${lib})
  elseif(MSVC)
    list(APPEND ALL_LIBRARIES -WHOLEARCHIVE:$<TARGET_FILE_NAME:${lib}>)
  else()
    list(APPEND ALL_LIBRARIES -Wl,--whole-archive ${lib} -Wl,--no-whole-archive)
  endif()
endmacro()

file(GLOB LIB_SOURCES "${PROJECT_SOURCE_DIR}/src/autoscore/*.cc")
list(APPEND LIB_SOURCES src/tinyxml2.cc)
add_library(auto-score ${LIB_SOURCES})
link_whole_archive(auto-score)
target_link_libraries(auto-score ${OpenCV_LIBS})

list(APPEND ALL_LIBRARIES auto-score)
message(STATUS ${ALL_LIBRARIES})

file(GLOB BIN_SOURCES "${PROJECT_SOURCE_DIR}/src/binaries/*.cc")
foreach(filename ${BIN_SOURCES})
  get_filename_component(name ${filename} NAME_WE)
  message(STATUS "Preparing ${name} binary")
  add_executable(${name} ${filename})
  target_link_libraries(${name} ${ALL_LIBRARIES} ${OpenCV_LIBS})
endforeach()
