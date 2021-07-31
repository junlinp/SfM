cmake_minimum_required(VERSION 3.20)
project(SfM)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
#find_package(Ceres REQUIRED)
find_package(cereal REQUIRED)
#find_package(glog REQUIRED)
find_package(GTest REQUIRED)
find_package(Eigen3 REQUIRED)

add_subdirectory(src/third_party)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
#add_executable(SfM main.cpp src/feature_extractor.cc src/feature_extractor.hpp  src/descriptor.cc   src/camera_model.hpp src/image.cc src/image.hpp)
#target_link_libraries(SfM ${OpenCV_LIBS} ${CERES_LIBRARIES})


add_library(SfMData src/sfm_data_io.cc src/sfm_data_io.hpp src/sfm_data.hpp)
target_link_libraries(SfMData PRIVATE cereal::cereal)

add_library(FeatureExtractor 
src/feature_extractor/vlfeat_feature_extractor.cc 
src/feature_extractor/opencv_orb_extractor.cc)
target_link_libraries(FeatureExtractor PUBLIC SfMData PRIVATE vlfeat ${OpenCV_LIBS})

add_library(Solver src/solver/fundamental_solver.cc src/solver/fundamental_solver.hpp)
target_include_directories(Solver PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(Solver PUBLIC Eigen3::Eigen)

target_include_directories(FeatureExtractor PRIVATE ${OpenCV_INCLUDE_DIRS})
add_executable(list_image src/list_image.cpp)
target_link_libraries(list_image PRIVATE SfMData)

add_executable(FeatureExtractor_main src/feature_extractor_main.cc)
target_link_libraries(FeatureExtractor_main PRIVATE FeatureExtractor)


add_library(FeatureMatch src/feature_matcher/brute_force_matcher.cc src/feature_matcher/exhaustive_pair_builder.cc)
target_include_directories(FeatureMatch PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(FeatureMatch PRIVATE SfMData)

add_executable(FeatureMatch_main src/feature_match_main.cc)
target_link_libraries(FeatureMatch_main PRIVATE FeatureMatch)

add_executable(SFM_main src/sfm_main.cc)
target_link_libraries(SFM_main PRIVATE SfMData Eigen3::Eigen Solver)

add_executable(DrawMatch_main src/draw_match_main.cc)
target_link_libraries(DrawMatch_main PUBLIC SfMData ${OpenCV_LIBS})

add_executable(Unittest src/unittest.cc)
target_link_libraries(Unittest PUBLIC GTest::gtest Solver)