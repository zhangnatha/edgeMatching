cmake_minimum_required(VERSION 3.16...3.31)
project(shapeMatch)

option(SHAPE_MATCH_VISUALIZE_COARSE "Show coarse matching candidates while inference runs" OFF)
option(SHAPE_MATCH_VISUALIZE_FINE "Show each coarse-to-fine refinement step" OFF)
option(BUILD_QT_CLIENT "Build the Qt5 Widgets client" ON)

# --- C++ 标准 ---
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
find_package(Threads REQUIRED)
find_package(OpenMP QUIET)

# --- OpenCV ---
# 优先使用用户显式指定的 OpenCV_DIR；否则使用仓库内的预编译依赖。
if(NOT OpenCV_DIR AND EXISTS "${CMAKE_SOURCE_DIR}/3rdparty/opencv/lib/cmake/opencv4")
    set(OpenCV_DIR "${CMAKE_SOURCE_DIR}/3rdparty/opencv/lib/cmake/opencv4")
endif()
find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc highgui imgcodecs calib3d)

# --- 头文件目录 ---
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

# --- 编译器选项 ---
# GCC/Clang 使用 OpenMP 和位置无关代码；MSVC 使用 CMake 默认 ABI，避免传入
# Unix 专用的 -fopenmp/-fPIC 参数。
if(MSVC)
    add_compile_options(/O2 /utf-8)
    add_definitions(-DNOMINMAX)
else()
    add_compile_options(-O3 -fPIC)
endif()

# --- 输出构建信息 ---
message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "OpenCV_LIBRARIES = ${OpenCV_LIBRARIES}")

# --- 构建共享库 ---
add_library(MakeTemplate SHARED
    src/MakeTemplateV1.cpp
)
target_link_libraries(MakeTemplate PUBLIC ${OpenCV_LIBRARIES} Threads::Threads)

add_library(FindTemplate SHARED
    src/FindTemplateV1.cpp
)
target_compile_definitions(FindTemplate PRIVATE
    SHAPE_MATCH_VISUALIZE_COARSE=$<BOOL:${SHAPE_MATCH_VISUALIZE_COARSE}>
    SHAPE_MATCH_VISUALIZE_FINE=$<BOOL:${SHAPE_MATCH_VISUALIZE_FINE}>
)
target_link_libraries(FindTemplate PUBLIC ${OpenCV_LIBRARIES} Threads::Threads)

# --- 构建可执行文件 ---
add_executable(train ${CMAKE_SOURCE_DIR}/train.cpp)
target_link_libraries(train PRIVATE MakeTemplate ${OpenCV_LIBRARIES})

add_executable(inference ${CMAKE_SOURCE_DIR}/inference.cpp)
target_link_libraries(inference PRIVATE FindTemplate ${OpenCV_LIBRARIES})

if(OpenMP_CXX_FOUND)
    target_link_libraries(MakeTemplate PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(FindTemplate PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(train PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(inference PRIVATE OpenMP::OpenMP_CXX)
endif()

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/DeployWindowsRuntime.cmake")
shape_match_deploy_windows_runtime(train)
shape_match_deploy_windows_runtime(inference)

# --- 安装库 ---
include(GNUInstallDirs)
install(TARGETS MakeTemplate FindTemplate
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS train inference
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# --- 安装公共头文件 ---
install(FILES
    include/MakeTemplateV1.h
    include/FindTemplateV1.h
    include/Type.h
    include/Timer.h
    include/ROI.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/shapeMatch
)

if(BUILD_QT_CLIENT)
    # Prefer the self-hosted Qt installation produced by UI/build_qt5.sh.
    # CMAKE_PREFIX_PATH is prepended so an explicitly supplied Qt toolchain
    # still remains available as a fallback when the local install is absent.
    set(_shape_match_qt5_prefix "${CMAKE_SOURCE_DIR}/3rdparty/qt5")
    if(NOT Qt5_DIR AND EXISTS "${_shape_match_qt5_prefix}/lib/cmake/Qt5/Qt5Config.cmake")
        list(PREPEND CMAKE_PREFIX_PATH "${_shape_match_qt5_prefix}")
        set(Qt5_DIR "${_shape_match_qt5_prefix}/lib/cmake/Qt5" CACHE PATH
            "Qt5 package configuration directory")
        message(STATUS "Using repository Qt5 installation: ${_shape_match_qt5_prefix}")
    endif()
    add_subdirectory(UI)
endif()

include(CTest)
if(BUILD_TESTING)
    find_package(Python3 QUIET COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
        add_test(NAME assert_matrix
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/scripts/verify_examples.py"
                --build-dir "$<TARGET_FILE_DIR:train>"
                --output-dir "${CMAKE_BINARY_DIR}/assert_matrix")
        set_tests_properties(assert_matrix PROPERTIES TIMEOUT 600)
    endif()
    if(TARGET shapeMatchQtClient)
        add_test(NAME qt_client_startup COMMAND shapeMatchQtClient --smoke-test)
        set_tests_properties(qt_client_startup PROPERTIES
            ENVIRONMENT "QT_QPA_PLATFORM=offscreen" TIMEOUT 30)
    endif()
endif()
