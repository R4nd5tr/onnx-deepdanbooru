cmake_minimum_required(VERSION 3.20)
project(autotagger 
        VERSION 0.0.0
        LANGUAGES CXX)


# setup configuration
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_BUILD_TYPE Debug)


set(PROJECT_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/cpp_deploy/include)
file(GLOB_RECURSE SRC_FILES ${PROJECT_SOURCE_DIR}/cpp_deploy/src/*.cpp)

add_library(autotagger SHARED ${SRC_FILES})
target_include_directories(autotagger PRIVATE ${PROJECT_INCLUDE_DIRS})
target_link_libraries(autotagger PRIVATE tensorflow-lite)