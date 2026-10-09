# Place runtime dependencies alongside executables for direct Windows launches.
function(shape_match_deploy_windows_runtime target)
    if(NOT WIN32)
        return()
    endif()
    set(opencv_runtime_targets ${OpenCV_LIBRARIES} ${OpenCV_LIB_COMPONENTS})
    list(REMOVE_DUPLICATES opencv_runtime_targets)
    foreach(dependency IN LISTS opencv_runtime_targets)
        if(TARGET "${dependency}")
            get_target_property(kind "${dependency}" TYPE)
            if(kind STREQUAL "SHARED_LIBRARY")
                add_custom_command(TARGET ${target} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "$<TARGET_FILE:${dependency}>" "$<TARGET_FILE_DIR:${target}>")
            endif()
        endif()
    endforeach()
    if(MINGW)
        foreach(name libgcc_s_seh-1.dll libstdc++-6.dll libwinpthread-1.dll libgomp-1.dll)
            execute_process(COMMAND "${CMAKE_CXX_COMPILER}" "-print-file-name=${name}"
                OUTPUT_VARIABLE runtime OUTPUT_STRIP_TRAILING_WHITESPACE)
            if(NOT EXISTS "${runtime}")
                get_filename_component(compiler_dir "${CMAKE_CXX_COMPILER}" DIRECTORY)
                set(runtime "${compiler_dir}/${name}")
            endif()
            if(EXISTS "${runtime}")
                add_custom_command(TARGET ${target} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${runtime}" "$<TARGET_FILE_DIR:${target}>")
            endif()
        endforeach()
    endif()
endfunction()
