cmake_minimum_required(VERSION 3.0.0)

project(kbestAssgnSLAM)

#SET(CMAKE_CXX_FLAGS "-g -std=c++17")
SET(CMAKE_CXX_FLAGS "-g -Ofast -march=native -std=c++17")

# Find GTSAM components
find_package(GTSAM REQUIRED) # Uses installed package
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Matplot++ REQUIRED)

include_directories(${GTSAM_INCLUDE_DIR})
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})


find_package(GTSAM_QUADRICS REQUIRED)
include_directories(${GTSAM_QUADRICS_INCLUDE_DIR})

# compile .exe files, exclude from all and build only if make examples
add_executable(semslamRun system.cpp bbNet.cpp slidingWindow.cpp shortestPathCPP.cpp nwPerm.cpp kittiReader.cpp dataframe.cpp assignment.cpp)
add_executable(compMethods comparison.cpp shortestPathCPP.cpp nwPerm.cpp assignment.cpp)


# link with gtsam (or recieve undefined refrences)
target_link_libraries(semslamRun gtsam ${EIGEN3_LIBS} gtsam_quadrics ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT} Matplot++::matplot)
target_link_libraries(compMethods gtsam ${EIGEN3_LIBS} gtsam_quadrics ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT} Matplot++::matplot)

