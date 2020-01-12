cmake_minimum_required(VERSION 3.11)


#コンパイラーがC++17に対応しているかチェック
include(CheckCXXCompilerFlag)

enable_language(CXX)

set(CMAKE_CXX_STANDARD 17)
CHECK_CXX_COMPILER_FLAG("-std=c++17" COMPILER_SUPPORTS_CXX17)
if(NOT ${COMPILER_SUPPORTS_CXX17})
    message(FATAL_ERROR "${CMAKE_CXX_COMPILER} doesn't support C++17")
endif()

set(CMAKE_CXX_FLAGS "-fmax-errors=10 -ftemplate-backtrace-limit=0 ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -g ${CMAKE_CXX_FLAGS_RELEASE}")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Og -g -D_GLIBCXX_DEBUG -DUSE_FULL_ASSERT")


# Set warning flags
set(CXX_WARNING_FLAGS
    -Wall
    -Wextra
    -Wconversion
    -Wswitch-default
    -Wdisabled-optimization
    -Wformat
    -Winit-self
    -Woverloaded-virtual
    -Wfloat-equal
    -Wno-old-style-cast
    -Wno-pragmas
    -Wno-ignored-attributes
    -Wreturn-type
    )
foreach(FLAG IN LISTS CXX_WARNING_FLAGS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FLAG}")
endforeach()


if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release)
endif()
message("Build type:\n\t${CMAKE_BUILD_TYPE}")


add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/external/eigen" "eigen")
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_definitions(eigen INTERFACE -DEIGEN_NO_DEBUG)
endif()


file(GLOB children RELATIVE "${CMAKE_CURRENT_LIST_DIR}" *)
foreach(child ${children})
    if(IS_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/${child}" AND EXISTS "${CMAKE_CURRENT_LIST_DIR}/${child}/CMakeLists.txt")
        add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/${child}" "${child}")
    endif()
endforeach()
