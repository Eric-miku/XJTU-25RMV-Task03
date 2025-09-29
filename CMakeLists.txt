cmake_minimum_required(VERSION 3.10)
project(CeresTest)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Ceres REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(include)


add_executable(ceres_test 
            src/main.cpp 
            src/BallDetector.cpp
            src/TrajectoryFitter.cpp)

# Ceres 1.x 用这种方式链接
include_directories(${CERES_INCLUDE_DIRS})
target_link_libraries(ceres_test ${CERES_LIBRARIES})
target_link_libraries(ceres_test ${OpenCV_LIBS})