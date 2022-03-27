cmake_minimum_required(VERSION 3.10)
project(SfM)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
find_package(cereal REQUIRED)
#find_package(glog REQUIRED)
find_package(GTest REQUIRED)
find_package(Eigen3 REQUIRED)
enable_testing()
MACRO(Unit_Test NAMESPACE NAME EXTRA_LIB PATH)
    add_executable(${NAMESPACE}_test_${NAME} ${PATH})
    target_link_libraries(${NAMESPACE}_test_${NAME} PUBLIC ${GTEST_LIBRARIES} ${EXTRA_LIB})

    add_test(${NAME} ${NAMESPACE}_test_${NAME} COMMAND $<TARGET_FILE:${NAMESPACE}_test_${NAME})
ENDMACRO(Unit_Test)

add_subdirectory(src/third_party)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

add_library(SfMData src/sfm_data_io.cc src/sfm_data_io.hpp src/sfm_data.hpp)
target_link_libraries(SfMData PRIVATE cereal::cereal PUBLIC Eigen3::Eigen)

add_library(FeatureExtractor 
src/feature_extractor/vlfeat_feature_extractor.cc 
src/feature_extractor/opencv_orb_extractor.cc)
target_link_libraries(FeatureExtractor PUBLIC SfMData PRIVATE vlfeat ${OpenCV_LIBS})

add_library(Solver
 src/solver/fundamental_solver.cc
 src/solver/fundamental_solver.hpp
 src/solver/triangular_solver.cc
 src/solver/trifocal_tensor_solver.cc
 src/solver/self_calibration_solver.cc
 src/solver/resection_solver.cc
 src/solver/epnp_solver.cc
 )
 
target_include_directories(Solver PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(Solver PUBLIC Eigen3::Eigen Ceres::ceres)
Unit_Test(Solver trifocal_tensor_solver Solver src/solver/trifocal_tensor_solver_test.cc)
Unit_Test(Solver self_calibration_solver Solver src/solver/self_calibration_solver_test.cc)
Unit_Test(Solver epnp_solver Solver src/solver/epnp_solver_test.cc)

target_include_directories(FeatureExtractor PRIVATE ${OpenCV_INCLUDE_DIRS})
add_executable(list_image src/list_image.cpp)
target_link_libraries(list_image PRIVATE SfMData)

add_executable(FeatureExtractor_main src/feature_extractor_main.cc)
target_link_libraries(FeatureExtractor_main PRIVATE FeatureExtractor)


add_library(FeatureMatch src/feature_matcher/brute_force_matcher.cc src/feature_matcher/exhaustive_pair_builder.cc src/feature_matcher/cascade_hash_matcher.cc)
target_include_directories(FeatureMatch PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(FeatureMatch PRIVATE SfMData)
Unit_Test(Matcher match FeatureMatch src/feature_matcher/match_test.cc)

add_executable(FeatureMatch_main src/feature_match_main.cc)
target_link_libraries(FeatureMatch_main PRIVATE FeatureMatch SfMData)

add_executable(SFM_main src/sfm_main.cc src/euclidean_constructure.cc)
target_link_libraries(SFM_main PRIVATE SfMData Eigen3::Eigen Solver)

add_executable(DrawMatch_main src/draw_match_main.cc)
target_link_libraries(DrawMatch_main PUBLIC SfMData ${OpenCV_LIBS})

add_executable(DrawMatchFilter_main src/draw_match_filter_main.cc)
target_link_libraries(DrawMatchFilter_main PUBLIC SfMData ${OpenCV_LIBS} Solver)

add_executable(Unittest src/unittest.cc)
target_link_libraries(Unittest PUBLIC Solver ${GTEST_LIBRARIES})
add_test(Name Unittest COMMAND ${CMAKE_CURRENT_BINARY_DIR}/Unittest)