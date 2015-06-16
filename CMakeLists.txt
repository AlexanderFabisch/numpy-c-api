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

file(COPY functions.py DESTINATION .)
file(COPY classes.py DESTINATION .)

add_executable(copy_numpy_array copy_numpy_array.cpp python_interpreter.cpp)
target_link_libraries(copy_numpy_array ${PYTHON_LIBRARIES})

add_executable(create_object create_object.cpp python_interpreter.cpp)
target_link_libraries(create_object ${PYTHON_LIBRARIES})

add_executable(hello_world hello_world.cpp python_interpreter.cpp)
target_link_libraries(hello_world ${PYTHON_LIBRARIES})

add_executable(import_error import_error.cpp python_interpreter.cpp)
target_link_libraries(import_error ${PYTHON_LIBRARIES})

add_executable(pass_pylist pass_pylist.cpp python_interpreter.cpp)
target_link_libraries(pass_pylist ${PYTHON_LIBRARIES})
