cmake_minimum_required(VERSION 3.18)
project(GenOCR CXX)

# Check that cpp_core was built
set(CPP_CORE_LIB ${CMAKE_CURRENT_SOURCE_DIR}/cpp_core/build/libgenocr_core.a)
if(NOT EXISTS ${CPP_CORE_LIB})
    message(FATAL_ERROR 
        "cpp_core not built! Run: \n"
        "cd cpp_core && mkdir build && cd build && cmake .. && make -j")
endif()

message(STATUS "Using pre-built cpp_core: ${CPP_CORE_LIB}")

# Only build Python bindings - cpp_core already built
option(BUILD_PYTHON_BINDINGS "Build Python bindings" OFF)

if(BUILD_PYTHON_BINDINGS)
    find_package(pybind11 QUIET)
    if(pybind11_FOUND)
        add_subdirectory(python_bindings)
    else()
        message(WARNING "pybind11 not found")
    endif()
endif()
