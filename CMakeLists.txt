cmake_minimum_required(VERSION 3.20)
project(autotagger 
        VERSION 0.0.0
        LANGUAGES CXX)


# setup configuration
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ONNXRUNTIME_DIR ${PROJECT_SOURCE_DIR}/cpp_deploy/external/onnxruntime-win-x64-gpu-1.23.1)
set(OPENCV_DIR ${PROJECT_SOURCE_DIR}/cpp_deploy/external/opencv/build)
set(HEADER_LIB_DIR ${PROJECT_SOURCE_DIR}/cpp_deploy/external/include)

set(OPENCV_DEBUG_LIB ${OPENCV_DIR}/x64/vc16/lib/opencv_world4120d.lib)
set(OPENCV_RELEASE_LIB ${OPENCV_DIR}/x64/vc16/lib/opencv_world4120.lib)

set(PROJECT_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/cpp_deploy/include)
set(SRC_FILE_DIRS ${PROJECT_SOURCE_DIR}/cpp_deploy/src)
set(ALL_INCLUDE_DIR ${PROJECT_INCLUDE_DIRS} ${ONNXRUNTIME_DIR}/include ${OPENCV_DIR}/include ${HEADER_LIB_DIR})
set(ALL_LIB ${ONNXRUNTIME_DIR}/lib/onnxruntime.lib "$<$<CONFIG:Debug>:${OPENCV_DEBUG_LIB}>" "$<$<CONFIG:Release>:${OPENCV_RELEASE_LIB}>"
)

add_library(autotagger_defalt SHARED ${SRC_FILE_DIRS}/defalt.cpp)
target_compile_definitions(autotagger_defalt PRIVATE AUTOTAGGER_EXPORTS)
target_include_directories(autotagger_defalt PRIVATE ${ALL_INCLUDE_DIR})
target_link_libraries(autotagger_defalt PRIVATE ${ALL_LIB})

if(BUILD_TEST_TARGET STREQUAL "ON")
    set(TEST_FILES ${PROJECT_SOURCE_DIR}/cpp_deploy/test.cpp) # include test file
    add_executable(TEST_autotagger
    ${SRC_FILE_DIRS}
    ${TEST_FILES}
    )
    target_compile_definitions(TEST_autotagger PRIVATE STATIC_TEST_BUILD)
    target_include_directories(TEST_autotagger PRIVATE ${ALL_INCLUDE_DIR})
    target_link_libraries(TEST_autotagger PRIVATE ${ALL_LIB})
endif()
