cmake_minimum_required(VERSION 3.20)
project(SfM)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
#find_package(Ceres REQUIRED)
find_package(cereal REQUIRED)
#find_package(glog REQUIRED)
find_package(GTest REQUIRED)

add_subdirectory(src/third_party)
#add_executable(SfM main.cpp src/feature_extractor.cc src/feature_extractor.hpp  src/descriptor.cc   src/camera_model.hpp src/image.cc src/image.hpp)
#target_link_libraries(SfM ${OpenCV_LIBS} ${CERES_LIBRARIES})


add_library(SfMData src/sfm_data_io.cc)
target_link_libraries(SfMData PRIVATE cereal::cereal)

add_library(FeatureExtractor src/feature_extractor/vlfeat_feature_extractor.cc)
target_link_libraries(FeatureExtractor PUBLIC SfMData PRIVATE vlfeat ${OpenCV_LIBS})

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




add_executable(Unittest src/unittest.cc)
target_link_libraries(Unittest PUBLIC GTest::gtest)