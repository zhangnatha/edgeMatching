cmake_minimum_required(VERSION 3.10)
project(shapeMatch)

# --- C++ standard ---
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- OpenCV ---
set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/3rdparty/opencv/lib/cmake/opencv4)
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui imgcodecs calib3d)

# --- Include directories ---
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

# --- Compiler flags ---
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -march=native -fopenmp -msse -msse2 -msse3 -msse4 -mavx -O3 -fPIC")

# --- Print info ---
message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "OpenCV_LIBRARIES = ${OpenCV_LIBRARIES}")

# --- Build shared libraries ---
add_library(MakeTemplate SHARED
    src/MakeTemplateV1.cpp
)
target_link_libraries(MakeTemplate ${OpenCV_LIBRARIES})

add_library(FindTemplate SHARED
    src/FindTemplateV1.cpp
)
target_link_libraries(FindTemplate ${OpenCV_LIBRARIES})

# --- Build executables ---
add_executable(train ${CMAKE_SOURCE_DIR}/train.cpp)
target_link_libraries(train MakeTemplate ${OpenCV_LIBRARIES})

add_executable(inference ${CMAKE_SOURCE_DIR}/inference.cpp)
target_link_libraries(inference FindTemplate ${OpenCV_LIBRARIES})

# --- Install libraries ---
install(TARGETS MakeTemplate FindTemplate
        LIBRARY DESTINATION ${CMAKE_BINARY_DIR}/publish/lib
)
install(TARGETS train inference
        RUNTIME DESTINATION ${CMAKE_BINARY_DIR}/publish/bin
)

# --- Install public headers ---
install(FILES
    include/MakeTemplateV1.h
    include/FindTemplateV1.h
    include/Type.h
    include/Timer.h
    include/ROI.h
    DESTINATION ${CMAKE_BINARY_DIR}/publish/include/shapeMatch
)

