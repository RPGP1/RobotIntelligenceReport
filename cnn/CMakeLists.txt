cmake_minimum_required(VERSION 3.11)

project(cnn CXX)

file(GLOB_RECURSE CNN_SOURCE "${CMAKE_CURRENT_LIST_DIR}/*.cpp")

add_executable(cnn ${CNN_SOURCE})
target_link_libraries(cnn stdc++fs eigen cmdline)
target_include_directories(cnn PRIVATE "${CMAKE_CURRENT_LIST_DIR}")

set_target_properties(cnn
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
