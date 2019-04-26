
cmake_minimum_required (VERSION 2.6)
project (colorize)

find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

set(NAME_SRC
    src/main.cpp
)

set(NAME_HEADERS
     # include/header.h
)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_CXX_EXTENSIONS OFF)

# set(CMAKE_BINARY_DIR )
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# message(${CMAKE_BINARY_DIR})

include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/include )

add_executable(colorize ${NAME_SRC} ${NAME_HEADERS})
target_link_libraries(colorize ${OpenCV_LIBS} Eigen3::Eigen)
