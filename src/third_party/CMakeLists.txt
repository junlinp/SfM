cmake_minimum_required(VERSION 3.10)

project(vlfeat C)

FILE(GLOB source_files vl/*.c)

add_library(${PROJECT_NAME} ${source_files})
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/vl)

