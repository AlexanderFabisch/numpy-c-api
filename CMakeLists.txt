cmake_minimum_required(VERSION 2.6)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
add_definitions(-DPYTHON_SUPPORT)
add_definitions(-DUSE_MEMORYVIEWS)
add_definitions(-std=c++11)
find_package(PythonLibs REQUIRED)
find_package(Numpy REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${NUMPY_INCLUDE_DIRS})
include_directories(.)
link_directories(${PYTHON_LIBRARY_DIRS})

add_executable(copy_numpy_array copy_numpy_array.cpp python_interpreter.cpp)
target_link_libraries(
    copy_numpy_array
    ${EXTRA_LIBRARIES}
    ${PYTHON_LIBRARIES})

file(COPY functions.py DESTINATION .)