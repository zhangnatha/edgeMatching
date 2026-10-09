# Inspect the executable and deployed Qt plugins, and copy their dependency closure.
if(NOT EXISTS "${EXECUTABLE}")
    message(FATAL_ERROR "Executable not found: ${EXECUTABLE}")
endif()
get_filename_component(destination "${EXECUTABLE}" DIRECTORY)
string(REPLACE "|" ";" search_dirs "${SEARCH_DIRS}")
if(WIN32)
    set(CMAKE_GET_RUNTIME_DEPENDENCIES_PLATFORM windows+pe)
    set(CMAKE_GET_RUNTIME_DEPENDENCIES_TOOL "${RUNTIME_TOOL}")
    set(CMAKE_GET_RUNTIME_DEPENDENCIES_COMMAND "${OBJDUMP}")
endif()
set(plugins)
if(WINDEPLOYQT)
    execute_process(COMMAND "${WINDEPLOYQT}" --no-translations "${EXECUTABLE}"
        RESULT_VARIABLE deploy_result)
    if(NOT deploy_result EQUAL 0)
        message(FATAL_ERROR "windeployqt failed: ${deploy_result}")
    endif()
    get_filename_component(qt_bin "${WINDEPLOYQT}" DIRECTORY)
    set(offscreen_src "${qt_bin}/../plugins/platforms/qoffscreen.dll")
    if(EXISTS "${offscreen_src}")
        execute_process(COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${offscreen_src}" "${destination}/platforms/qoffscreen.dll")
    endif()
    foreach(plugin_dir platforms imageformats iconengines styles)
        if(EXISTS "${destination}/${plugin_dir}")
            file(GLOB_RECURSE dir_plugins "${destination}/${plugin_dir}/*.dll")
            list(APPEND plugins ${dir_plugins})
        endif()
    endforeach()
endif()
set(system_excludes)
if(WIN32)
    file(TO_CMAKE_PATH "$ENV{SystemRoot}" system_root)
    set(system_pattern "")
    string(LENGTH "${system_root}" system_length)
    math(EXPR last_index "${system_length} - 1")
    foreach(index RANGE 0 ${last_index})
        string(SUBSTRING "${system_root}" ${index} 1 character)
        if(character MATCHES "[A-Za-z]")
            string(TOLOWER "${character}" lower)
            string(TOUPPER "${character}" upper)
            string(APPEND system_pattern "[${lower}${upper}]")
        else()
            string(APPEND system_pattern "${character}")
        endif()
    endforeach()
    list(APPEND system_excludes "^${system_pattern}/.*")
endif()
file(GET_RUNTIME_DEPENDENCIES
    EXECUTABLES "${EXECUTABLE}"
    LIBRARIES ${plugins}
    DIRECTORIES ${search_dirs}
    PRE_EXCLUDE_REGEXES
        "[Aa][Pp][Ii]-[Mm][Ss]-.*"
        "[Ee][Xx][Tt]-[Mm][Ss]-.*"
        "^[Kk][Ee][Rr][Nn][Ee][Ll]32\\.[Dd][Ll][Ll]$"
        "^[Uu][Ss][Ee][Rr]32\\.[Dd][Ll][Ll]$"
        "^[Gg][Dd][Ii]32\\.[Dd][Ll][Ll]$"
        "^[Aa][Dd][Vv][Aa][Pp][Ii]32\\.[Dd][Ll][Ll]$"
        "^[Ss][Hh][Ee][Ll][Ll]32\\.[Dd][Ll][Ll]$"
        "^[Oo][Ll][Ee]32\\.[Dd][Ll][Ll]$"
        "^[Oo][Ll][Ee][Aa][Uu][Tt]32\\.[Dd][Ll][Ll]$"
        "^[Ww][Ss]2_32\\.[Dd][Ll][Ll]$"
        "^[Nn][Tt][Dd][Ll][Ll]\\.[Dd][Ll][Ll]$"
    POST_EXCLUDE_REGEXES ${system_excludes}
    RESOLVED_DEPENDENCIES_VAR resolved
    UNRESOLVED_DEPENDENCIES_VAR unresolved
    CONFLICTING_DEPENDENCIES_PREFIX conflicts)
if(unresolved)
    message(FATAL_ERROR "Missing runtime DLLs for ${EXECUTABLE}: ${unresolved}. Search directories: ${search_dirs}")
endif()
set(real_conflicts)
foreach(filename IN LISTS conflicts_FILENAMES)
    set(var_name "conflicts_${filename}")
    set(distinct_outside_dest)
    foreach(candidate IN LISTS ${var_name})
        get_filename_component(cand_dir "${candidate}" DIRECTORY)
        file(TO_CMAKE_PATH "${cand_dir}" cand_dir)
        file(TO_CMAKE_PATH "${destination}" dest_norm)
        if(WIN32 AND system_pattern AND candidate MATCHES "^${system_pattern}/.*")
            continue()
        endif()
        if(cand_dir STREQUAL dest_norm)
            continue()
        endif()
        list(APPEND distinct_outside_dest "${cand_dir}")
    endforeach()
    list(REMOVE_DUPLICATES distinct_outside_dest)
    list(LENGTH distinct_outside_dest outside_count)
    if(outside_count GREATER 1)
        list(APPEND real_conflicts "${filename}")
    endif()
endforeach()
if(real_conflicts)
    message(FATAL_ERROR "Conflicting runtime DLLs: ${real_conflicts}. Use the same compiler for Qt, OpenCV and the application.")
endif()
foreach(dependency IN LISTS resolved)
    get_filename_component(directory "${dependency}" DIRECTORY)
    if(NOT directory STREQUAL destination)
        execute_process(COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${dependency}" "${destination}" RESULT_VARIABLE copy_result)
        if(NOT copy_result EQUAL 0)
            message(FATAL_ERROR "Cannot deploy ${dependency}")
        endif()
    endif()
endforeach()
message(STATUS "Runtime dependencies deployed: ${EXECUTABLE}")
