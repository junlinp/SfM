cmake_minimum_required(VERSION 3.0)
project(SfM)

set(CMAKE_CXX_STANDARD 14)
include_directories(/usr/local/Cellar/opencv/4.1.0_2/include/opencv4)
include_directories(src)
include_directories(/usr/local/include)
link_directories(/usr/local/lib)

find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)


add_executable(SfM main.cpp src/feature_extractor.cc src/feature_extractor.hpp src/database.cc src/database.hpp src/unittest.hpp src/descriptor.cc src/descriptor.hpp src/keypoint.hpp src/camera_model.hpp src/image.cc src/image.hpp)
target_link_libraries(SfM ${OpenCV_LIBS} ${CERES_LIBRARIES} gtest sqlite3 )

add_executable(SfM_unittest src/unittest.cc src/feature_extractor.cc src/feature_extractor.hpp src/database.cc src/database.hpp src/unittest.hpp src/descriptor.cc src/descriptor.hpp src/keypoint.hpp src/camera_model.hpp src/image.cc src/image.hpp)
target_link_libraries(SfM_unittest ${OpenCV_LIBS} ${CERES_LIBRARIES} gtest sqlite3 )

