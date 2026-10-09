cmake_minimum_required(VERSION 3.16...3.31)

# The supported build is from the repository root.  Keeping this file small
# and self-contained makes the UI target easy to inspect and reuse in IDEs.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(SHAPE_MATCH_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")
    set(_shape_match_qt5_prefix "${SHAPE_MATCH_ROOT_DIR}/3rdparty/qt5")
    if(NOT Qt5_DIR AND EXISTS "${_shape_match_qt5_prefix}/lib/cmake/Qt5/Qt5Config.cmake")
        list(PREPEND CMAKE_PREFIX_PATH "${_shape_match_qt5_prefix}")
        set(Qt5_DIR "${_shape_match_qt5_prefix}/lib/cmake/Qt5" CACHE PATH
            "Qt5 package configuration directory")
    endif()
    project(shapeMatchQtClient LANGUAGES CXX)
    set(CMAKE_CXX_STANDARD 11)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    find_package(Threads REQUIRED)
    find_package(OpenMP QUIET)
    if(MSVC)
        add_compile_options(/utf-8)
        add_definitions(-DNOMINMAX)
    endif()
    if(NOT OpenCV_DIR AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/opencv/lib/cmake/opencv4")
        set(OpenCV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/opencv/lib/cmake/opencv4")
    endif()
    find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc highgui imgcodecs calib3d)
    find_package(Qt5 5.9 REQUIRED COMPONENTS Widgets Concurrent)
    add_library(MakeTemplate SHARED ${CMAKE_CURRENT_SOURCE_DIR}/../src/MakeTemplateV1.cpp)
    add_library(FindTemplate SHARED ${CMAKE_CURRENT_SOURCE_DIR}/../src/FindTemplateV1.cpp)
    foreach(core_target MakeTemplate FindTemplate)
        target_include_directories(${core_target} PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/../include ${OpenCV_INCLUDE_DIRS})
        target_link_libraries(${core_target} PUBLIC ${OpenCV_LIBRARIES} Threads::Threads)
        if(OpenMP_CXX_FOUND)
            target_link_libraries(${core_target} PUBLIC OpenMP::OpenMP_CXX)
        endif()
    endforeach()
endif()

include(GNUInstallDirs)

if(NOT DEFINED SHAPE_MATCH_ROOT_DIR)
    set(SHAPE_MATCH_ROOT_DIR "${CMAKE_SOURCE_DIR}")
endif()

find_package(Qt5 5.9 REQUIRED COMPONENTS Widgets Concurrent)
message(STATUS "Qt5 ${Qt5Core_VERSION_STRING}: ${Qt5_DIR}")

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

set(SHAPE_MATCH_TRANSLATION_TS ${CMAKE_CURRENT_SOURCE_DIR}/shape_match_en.ts)
get_target_property(_shape_match_qmake Qt5::qmake IMPORTED_LOCATION)
get_filename_component(_shape_match_qt_bin "${_shape_match_qmake}" DIRECTORY)
unset(SHAPE_MATCH_LRELEASE CACHE)
find_program(SHAPE_MATCH_LRELEASE NAMES lrelease lrelease-qt5
    HINTS "${_shape_match_qt_bin}" NO_DEFAULT_PATH)
if(SHAPE_MATCH_LRELEASE)
    set(SHAPE_MATCH_TRANSLATION_QM ${CMAKE_CURRENT_BINARY_DIR}/shape_match_en.qm)
    add_custom_command(OUTPUT ${SHAPE_MATCH_TRANSLATION_QM}
        COMMAND ${SHAPE_MATCH_LRELEASE} ${SHAPE_MATCH_TRANSLATION_TS}
                -qm ${SHAPE_MATCH_TRANSLATION_QM}
        DEPENDS ${SHAPE_MATCH_TRANSLATION_TS}
        COMMENT "Compiling Qt translations")
    add_custom_target(shape_match_translations DEPENDS ${SHAPE_MATCH_TRANSLATION_QM})
else()
    message(FATAL_ERROR "lrelease not found in ${_shape_match_qt_bin}; install Qt Linguist tools or rebuild local Qt5 with UI/build_qt5.sh")
endif()

add_executable(shapeMatchQtClient
    main.cpp
    MainWindow.cpp
    MainWindow.h
    ContourBuilder.cpp
    ContourBuilder.h
    MatchOverlay.cpp
    MatchOverlay.h
    ImageView.cpp
    ImageView.h
)
if(TARGET shape_match_translations)
    add_dependencies(shapeMatchQtClient shape_match_translations)
endif()

target_include_directories(shapeMatchQtClient PRIVATE
    ${SHAPE_MATCH_ROOT_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(shapeMatchQtClient PRIVATE
    MakeTemplate FindTemplate
    ${OpenCV_LIBRARIES}
    Qt5::Widgets Qt5::Concurrent
)

if(SHAPE_MATCH_TRANSLATION_QM)
    add_custom_command(TARGET shapeMatchQtClient POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${SHAPE_MATCH_TRANSLATION_QM}
                $<TARGET_FILE_DIR:shapeMatchQtClient>/shape_match_en.qm
    )
    install(FILES ${SHAPE_MATCH_TRANSLATION_QM}
        DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

set_target_properties(shapeMatchQtClient PROPERTIES
    OUTPUT_NAME shape_match_qt
)
include("${SHAPE_MATCH_ROOT_DIR}/cmake/DeployWindowsRuntime.cmake")
if(WIN32)
    find_program(SHAPE_MATCH_WINDEPLOYQT NAMES windeployqt
        HINTS "${_shape_match_qt_bin}" NO_DEFAULT_PATH)
    if(NOT SHAPE_MATCH_WINDEPLOYQT)
        message(FATAL_ERROR "windeployqt not found in selected Qt: ${_shape_match_qt_bin}")
    endif()
    add_custom_command(TARGET shapeMatchQtClient POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:MakeTemplate>" "$<TARGET_FILE:FindTemplate>"
            "$<TARGET_FILE_DIR:shapeMatchQtClient>")
endif()

shape_match_deploy_windows_runtime(shapeMatchQtClient)

install(TARGETS shapeMatchQtClient
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
