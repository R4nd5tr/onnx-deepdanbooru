cmake_minimum_required(VERSION 3.20)
project(autotagger 
        VERSION 0.0.0
        LANGUAGES CXX)


# setup configuration
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_BUILD_TYPE Debug)

set(ONNXRUNTIME_DIR ${PROJECT_SOURCE_DIR}/cpp_deploy/external/onnxruntime-win-x64-1.23.1)
set(OPENCV_DIR ${PROJECT_SOURCE_DIR}/cpp_deploy/external/opencv/build)


set(PROJECT_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/cpp_deploy/include)
file(GLOB_RECURSE SRC_FILES ${PROJECT_SOURCE_DIR}/cpp_deploy/src/*.cpp)
set(ALL_INCLUDE_DIR ${PROJECT_INCLUDE_DIRS} ${ONNXRUNTIME_DIR}/include ${OPENCV_DIR}/include)
set(ALL_LIB ${ONNXRUNTIME_DIR}/lib/onnxruntime.lib ${OPENCV_DIR}/x64/vc16/lib/opencv_world4120d.lib)

add_library(autotagger SHARED ${SRC_FILES})
target_include_directories(autotagger PRIVATE ${ALL_INCLUDE_DIR})
target_link_libraries(autotagger PRIVATE ${ALL_LIB})

if(BUILD_TEST_TARGET STREQUAL "ON")
    set(TEST_FILES ${PROJECT_SOURCE_DIR}/cpp_deploy/test.cpp) # include test files
    add_executable(TEST_autotagger
        ${SRC_FILES}
        ${TEST_FILES}
    )
    target_include_directories(TEST_autotagger PRIVATE ${ALL_INCLUDE_DIR})
    target_link_libraries(TEST_autotagger PRIVATE ${ALL_LIB})
endif()
