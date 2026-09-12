cmake_minimum_required(VERSION 3.10)
project(shapeMatch)

include(CTest)

option(SHAPE_MATCH_VISUALIZE_COARSE "Show coarse matching candidates while inference runs" OFF)
option(SHAPE_MATCH_VISUALIZE_FINE "Show each coarse-to-fine refinement step" OFF)
option(BUILD_QT_CLIENT "Build the Qt5 Widgets client" ON)

# --- C++ 标准 ---
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- OpenCV ---
set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/3rdparty/opencv/lib/cmake/opencv4)
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui imgcodecs calib3d)

# --- 头文件目录 ---
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

# --- 编译器选项 ---
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp -O3 -fPIC")

# --- 输出构建信息 ---
message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "OpenCV_LIBRARIES = ${OpenCV_LIBRARIES}")

# --- 构建共享库 ---
add_library(MakeTemplate SHARED
    src/MakeTemplateV1.cpp
)
target_link_libraries(MakeTemplate ${OpenCV_LIBRARIES})

add_library(FindTemplate SHARED
    src/FindTemplateV1.cpp
)
target_compile_definitions(FindTemplate PRIVATE
    SHAPE_MATCH_VISUALIZE_COARSE=$<BOOL:${SHAPE_MATCH_VISUALIZE_COARSE}>
    SHAPE_MATCH_VISUALIZE_FINE=$<BOOL:${SHAPE_MATCH_VISUALIZE_FINE}>
)
target_link_libraries(FindTemplate ${OpenCV_LIBRARIES})

# --- 构建可执行文件 ---
add_executable(train ${CMAKE_SOURCE_DIR}/train.cpp)
target_link_libraries(train MakeTemplate ${OpenCV_LIBRARIES})

add_executable(inference ${CMAKE_SOURCE_DIR}/inference.cpp)
target_link_libraries(inference FindTemplate ${OpenCV_LIBRARIES})

if(BUILD_TESTING)
    add_executable(test_matching tests/test_matching.cpp)
    target_link_libraries(test_matching MakeTemplate FindTemplate ${OpenCV_LIBRARIES})
    add_executable(test_intermediate_threshold tests/test_intermediate_threshold.cpp)
    target_compile_definitions(test_intermediate_threshold PRIVATE
        ASSERT_DIR="${CMAKE_SOURCE_DIR}/assert")
    target_link_libraries(test_intermediate_threshold MakeTemplate FindTemplate ${OpenCV_LIBRARIES})
    add_executable(test_devernay tests/test_devernay.cpp)
    target_link_libraries(test_devernay MakeTemplate ${OpenCV_LIBRARIES})
    add_executable(benchmark_matching tests/benchmark_matching.cpp)
    target_link_libraries(benchmark_matching FindTemplate ${OpenCV_LIBRARIES})
    add_test(NAME matching_regression COMMAND test_matching)
    add_test(NAME intermediate_threshold_regression COMMAND test_intermediate_threshold)
    add_test(NAME devernay_edge_regression COMMAND test_devernay)
    add_test(NAME cli_reject_negative_contrast COMMAND inference --min-contrast -1)
    add_test(NAME cli_reject_large_contrast COMMAND inference --min-contrast 362)
    add_test(NAME cli_reject_fractional_contrast COMMAND inference --min-contrast 1.5)
    add_test(NAME cli_reject_missing_contrast COMMAND inference --min-contrast)
    add_test(NAME cli_reject_bad_metric COMMAND inference --metric bad)
    set_tests_properties(cli_reject_negative_contrast cli_reject_large_contrast
                         cli_reject_fractional_contrast cli_reject_missing_contrast
                         cli_reject_bad_metric PROPERTIES WILL_FAIL TRUE)
endif()

# --- 安装库 ---
install(TARGETS MakeTemplate FindTemplate
        LIBRARY DESTINATION ${CMAKE_BINARY_DIR}/publish/lib
)
install(TARGETS train inference
        RUNTIME DESTINATION ${CMAKE_BINARY_DIR}/publish/bin
)

# --- 安装公共头文件 ---
install(FILES
    include/MakeTemplateV1.h
    include/FindTemplateV1.h
    include/Type.h
    include/Timer.h
    include/ROI.h
    DESTINATION ${CMAKE_BINARY_DIR}/publish/include/shapeMatch
)

if(BUILD_QT_CLIENT)
    # Prefer the self-hosted Qt installation produced by UI/build_qt5.sh.
    # CMAKE_PREFIX_PATH is prepended so an explicitly supplied Qt toolchain
    # still remains available as a fallback when the local install is absent.
    set(_shape_match_qt5_prefix "${CMAKE_SOURCE_DIR}/3rdparty/qt5")
    if(EXISTS "${_shape_match_qt5_prefix}/lib/cmake/Qt5/Qt5Config.cmake")
        list(PREPEND CMAKE_PREFIX_PATH "${_shape_match_qt5_prefix}")
        set(Qt5_DIR "${_shape_match_qt5_prefix}/lib/cmake/Qt5" CACHE PATH
            "Qt5 package configuration directory" FORCE)
        message(STATUS "Using repository Qt5 installation: ${_shape_match_qt5_prefix}")
    endif()
    add_subdirectory(UI)
endif()
