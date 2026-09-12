cmake_minimum_required(VERSION 3.10)

# The supported build is from the repository root.  Keeping this file small
# and self-contained makes the UI target easy to inspect and reuse in IDEs.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(SHAPE_MATCH_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")
    set(_shape_match_qt5_prefix "${SHAPE_MATCH_ROOT_DIR}/3rdparty/qt5")
    if(EXISTS "${_shape_match_qt5_prefix}/lib/cmake/Qt5/Qt5Config.cmake")
        list(PREPEND CMAKE_PREFIX_PATH "${_shape_match_qt5_prefix}")
        set(Qt5_DIR "${_shape_match_qt5_prefix}/lib/cmake/Qt5" CACHE PATH
            "Qt5 package configuration directory" FORCE)
    endif()
    project(shapeMatchQtClient LANGUAGES CXX)
    set(CMAKE_CXX_STANDARD 11)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/opencv/lib/cmake/opencv4")
        set(OpenCV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/opencv/lib/cmake/opencv4")
    endif()
    find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
    find_package(Qt5 REQUIRED COMPONENTS Widgets Concurrent)
    add_library(MakeTemplate SHARED ${CMAKE_CURRENT_SOURCE_DIR}/../src/MakeTemplateV1.cpp)
    add_library(FindTemplate SHARED ${CMAKE_CURRENT_SOURCE_DIR}/../src/FindTemplateV1.cpp)
    foreach(core_target MakeTemplate FindTemplate)
        target_include_directories(${core_target} PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/../include ${OpenCV_INCLUDE_DIRS})
        target_link_libraries(${core_target} PUBLIC ${OpenCV_LIBRARIES})
    endforeach()
endif()

if(NOT DEFINED SHAPE_MATCH_ROOT_DIR)
    set(SHAPE_MATCH_ROOT_DIR "${CMAKE_SOURCE_DIR}")
endif()

find_package(Qt5 REQUIRED COMPONENTS Widgets Concurrent)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

set(SHAPE_MATCH_TRANSLATION_TS ${CMAKE_CURRENT_SOURCE_DIR}/shape_match_en.ts)
find_program(SHAPE_MATCH_LRELEASE NAMES lrelease lrelease-qt5)
if(SHAPE_MATCH_LRELEASE)
    set(SHAPE_MATCH_TRANSLATION_QM ${CMAKE_CURRENT_BINARY_DIR}/shape_match_en.qm)
    add_custom_command(OUTPUT ${SHAPE_MATCH_TRANSLATION_QM}
        COMMAND ${SHAPE_MATCH_LRELEASE} ${SHAPE_MATCH_TRANSLATION_TS}
                -qm ${SHAPE_MATCH_TRANSLATION_QM}
        DEPENDS ${SHAPE_MATCH_TRANSLATION_TS}
        COMMENT "Compiling Qt translations")
    add_custom_target(shape_match_translations DEPENDS ${SHAPE_MATCH_TRANSLATION_QM})
else()
    message(WARNING "lrelease not found; the English translation catalog will not be built")
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
endif()

if(BUILD_TESTING)
    add_executable(test_ui_contours
        ../tests/test_ui_contours.cpp
        ContourBuilder.cpp
        ContourBuilder.h
        MatchOverlay.cpp
        MatchOverlay.h
    )
    target_include_directories(test_ui_contours PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${SHAPE_MATCH_ROOT_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
    )
    target_link_libraries(test_ui_contours PRIVATE
        MakeTemplate
        FindTemplate
        ${OpenCV_LIBRARIES}
        Qt5::Widgets
    )
    add_test(NAME ui_contour_dedup COMMAND test_ui_contours)
endif()

set_target_properties(shapeMatchQtClient PROPERTIES
    OUTPUT_NAME shape_match_qt
)
