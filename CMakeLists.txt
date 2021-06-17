cmake_minimum_required(VERSION 3.15)
project(SfM)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
find_package(cereal REQUIRED)
find_package(glog REQUIRED)

add_subdirectory(src/third_party)
#add_executable(SfM main.cpp src/feature_extractor.cc src/feature_extractor.hpp  src/descriptor.cc   src/camera_model.hpp src/image.cc src/image.hpp)
#target_link_libraries(SfM ${OpenCV_LIBS} ${CERES_LIBRARIES})

#add_executable(SfM_unittest src/unittest.cc src/feature_extractor.cc src/feature_extractor.hpp src/database.cc src/database.hpp src/unittest.hpp src/descriptor.cc src/descriptor.hpp src/keypoint.hpp src/camera_model.hpp src/image.cc src/image.hpp)
#target_link_libraries(SfM_unittest ${OpenCV_LIBS} ${CERES_LIBRARIES} gtest sqlite3 )

add_library(SfMData src/sfm_data_io.cc)
target_link_libraries(SfMData PRIVATE cereal)

add_library(FeatureExtractor src/feature_extractor/vlfeat_feature_extractor.cc)
target_link_libraries(FeatureExtractor PUBLIC SfMData PRIVATE vlfeat ${OpenCV_LIBS})
target_include_directories(FeatureExtractor PRIVATE ${OpenCV_INCLUDE_DIRS})

add_executable(list_image src/list_image.cpp)
target_link_libraries(list_image PRIVATE SfMData)

