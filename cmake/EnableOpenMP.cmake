# Let the GCC driver select its matching OpenMP runtime. Compile and link flags
# must stay together, including on MinGW installations whose paths contain spaces.
function(shape_match_enable_openmp target)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT CMAKE_DISABLE_FIND_PACKAGE_OpenMP)
        target_compile_options(${target} PUBLIC -fopenmp)
        target_link_options(${target} PUBLIC -fopenmp)
    elseif(OpenMP_CXX_FOUND)
        target_link_libraries(${target} PUBLIC OpenMP::OpenMP_CXX)
    endif()
endfunction()
